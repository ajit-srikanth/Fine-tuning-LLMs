from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
import wandb
from tqdm import tqdm
import argparse


class CNNDailyMailDataset(Dataset):
    def __init__(self, split, tokenizer, max_length, summary_length):
        self.data = pd.read_csv(f"./cnn_dailymail/{split}.csv")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summary_length = summary_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data.iloc[idx]['article']
        highlights = self.data.iloc[idx]['highlights']

        inputs = self.tokenizer(article, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer(highlights, max_length=self.summary_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


class SoftPromptTuning:
    def __init__(self, model_name="EleutherAI/gpt-neo-125m", initial_prompt="SUMMARIZE"):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.model = GPTNeoForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.initial_prompt_ids = self.tokenizer.encode(initial_prompt, add_special_tokens=False)
        self.num_prompts = len(self.initial_prompt_ids)
        self.embedding_size = self.model.config.hidden_size
        self.soft_prompts = nn.Embedding(self.num_prompts, self.embedding_size).to(self.device)        
        self.gpt_forward = self.model.forward
        self.initialize_soft_prompt()
        # self.model.forward = self.model_forward_with_soft_prompts
        
        # does not freeze the soft prompt layer
        self.freeze_model_parameters()
    
    def initialize_soft_prompt(self):
        with torch.no_grad():
            self.soft_prompts.weight.data = self.model.transformer.wte.weight[self.initial_prompt_ids].clone()

    def model_forward_with_soft_prompts(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        soft_prompt_embeds = self.soft_prompts(torch.arange(self.num_prompts).to(input_ids.device))
        
        soft_prompt_embeds = soft_prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        # print(soft_prompt_embeds)
        # exit(0)
        
        inputs_embeds = self.model.transformer.wte(input_ids)
        combined_embeds = torch.cat([soft_prompt_embeds, inputs_embeds], dim=1)
        combined_embeds = combined_embeds[:, :inputs_embeds.shape[1], :]
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(batch_size, self.num_prompts).to(attention_mask.device)
            combined_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
            combined_attention_mask = combined_attention_mask[:, :inputs_embeds.shape[1]]
        else:
            combined_attention_mask = None
        
        outputs = self.gpt_forward(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask, labels = labels)
        
        return outputs

    def freeze_model_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.soft_prompts.weight.requires_grad = True

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_soft_prompts(self):
        return self.soft_prompts



def train(spt_model, train_loader, optimizer, device):
    spt_model.model.train()
    total_loss = 0
    iii = 0
    # criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tokenizer.pad_token)
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = spt_model.model_forward_with_soft_prompts(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
        
        loss = outputs.loss
        # loss = criterion(labels, outputs)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(spt_model.soft_prompts.parameters(), clip_value=1)
        optimizer.step()
        iii += 1
        if iii > (len(train_loader)/10):
            break

    return total_loss / len(train_loader) * 10

def evaluate(spt_model, eval_loader, device):
    spt_model.model.eval()
    total_loss = 0
    iii = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = spt_model.model_forward_with_soft_prompts(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            iii += 1
            if iii > 500:
                break

    return total_loss / len(eval_loader) * len(eval_loader)/500

def calculate_rouge(spt_model, test_loader, tokenizer, SUMMARY_LENGTH, device):
    spt_model.model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating ROUGE"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            generated_ids = spt_model.model_forward_with_soft_prompts(input_ids=input_ids, attention_mask=attention_mask)
            generated_summaries = tokenizer.batch_decode(torch.argmax(generated_ids.logits, dim=-1), skip_special_tokens=True)
            reference_summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for gen, ref in zip(generated_summaries, reference_summaries):
                scores = scorer.score(ref, gen)
                for key in rouge_scores:
                    rouge_scores[key] += scores[key].fmeasure
                count += 1
            if count > 1000:
                break
    print(generated_summaries)
    print("*******************")
    print(reference_summaries)
    for key in rouge_scores:
        rouge_scores[key] /= count

    return rouge_scores


def main(args):
    enable_wandb = True
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = args.max_length
    model_name = args.model_name
    initial_prompt = args.initial_prompt

    soft_prompt_tuning = SoftPromptTuning(model_name=model_name, initial_prompt=initial_prompt)

#    soft_prompt_tuning.soft_prompts.load_state_dict(torch.load(f"./spt_models/model_epoch_3.pt"))
    
    # model = soft_prompt_tuning.get_model()
    tokenizer = soft_prompt_tuning.get_tokenizer()
    if enable_wandb:
        wandb.init(project="soft_prompt_tuning")
        wandb.config.update(args)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    
    train_dataset = CNNDailyMailDataset("train", tokenizer, MAX_LENGTH, MAX_LENGTH)
    val_dataset = CNNDailyMailDataset("validation", tokenizer, MAX_LENGTH, MAX_LENGTH)
    test_dataset = CNNDailyMailDataset("test", tokenizer, MAX_LENGTH, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # optimizer = torch.optim.AdamW(soft_prompt_tuning.model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(soft_prompt_tuning.soft_prompts.parameters(), lr=LEARNING_RATE)
    

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(soft_prompt_tuning, train_loader, optimizer, device)
        eval_loss = evaluate(soft_prompt_tuning, val_loader, device)
        if enable_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss
            })
        rouge_scores = calculate_rouge(soft_prompt_tuning, test_loader, tokenizer, MAX_LENGTH, device)
        print(f"ROUGE Scores: {rouge_scores}")
        # exit(0) 
        if enable_wandb:                                                                                                                                                                                wandb.log({                                                                                                                                                                                     "rouge1": rouge_scores['rouge1'],                                                                                                                                                           "rouge2": rouge_scores['rouge2'],                                                                                                                                                           "rougeL": rouge_scores['rougeL']                                                                                                                                                        })
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}")
        ## torch.save(model.state_dict(), f"./models/model_epoch_{epoch + 1}.pt")
        torch.save(soft_prompt_tuning.soft_prompts.state_dict(), f"./spt_models/model_epoch_{epoch + 1}.pt")
         
        # soft_prompt_tuning.soft_prompts.load_state_dict(torch.load(f"./spt_models/model_epoch_{epoch + 1}.pt"))


    # Evaluate on test set
    test_loss = evaluate(soft_prompt_tuning, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    rouge_scores = calculate_rouge(soft_prompt_tuning, test_loader, tokenizer, MAX_LENGTH, device)
    print(f"ROUGE Scores: {rouge_scores}")
    # exit(0)
    if enable_wandb:
        wandb.log({
            "test_loss": test_loss,
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL']
        })


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", help="Name of the model to use")
    Parser.add_argument("--initial_prompt", type=str, default="[SUMMARIZE]", help="Initial prompt for soft tuning")
    Parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    Parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    Parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    Parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the input")
    args = Parser.parse_args()
    main(args)

