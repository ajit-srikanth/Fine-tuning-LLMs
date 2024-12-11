from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import wandb
from rouge_score import rouge_scorer


model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m")

# Freeze most of the model's parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers (e.g., last 2 layers)
#for param in model.transformer.h[-2:].parameters():
#    param.requires_grad = True

# Unfreeze the output layer
for param in model.lm_head.parameters():
    param.requires_grad = True

class CNNDailyMailDataset(Dataset):
    def __init__(self, split, tokenizer, max_length):
        self.data = pd.read_csv(f"./cnn_dailymail/{split}.csv")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data.iloc[idx]['article']
        highlights = self.data.iloc[idx]['highlights']

        inputs = self.tokenizer(article, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer(highlights, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    iii = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()
        iii += 1
        if iii > (len(train_loader)/10):
            break

    return total_loss / len(train_loader) * len(train_loader) / iii

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    iii = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            iii += 1
            if iii > 500:
                break
    
    return total_loss / len(eval_loader) * len(eval_loader) / iii

def calculate_rouge(model, test_loader, tokenizer, summary_length, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating ROUGE"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

#            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=summary_length)
            generated_ids = model.generate(input_ids=input_ids[:,:summary_length-1], attention_mask=attention_mask[:,:summary_length-1], max_length=summary_length)
            generated_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)
#            print(generated_summaries)
#            print("*******************")
#            print(reference_summaries)
#            exit(0)

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


def main():

    wandb.init(project="traditional_fine_tuning_summarization")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 512
    

    # Log hyperparameters to wandb
    wandb.config.update({
        "model_name": "EleutherAI/gpt-neo-125m",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
    })

    
    train_dataset = CNNDailyMailDataset("train", tokenizer, MAX_LENGTH)
    val_dataset = CNNDailyMailDataset("validation", tokenizer, MAX_LENGTH)
    test_dataset = CNNDailyMailDataset("test", tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    prev_loss = 1000000 
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer, device)
        eval_loss = evaluate(model, val_loader, device)
        
        rouge_scores = calculate_rouge(model, test_loader, tokenizer, MAX_LENGTH, device)
        print(f"ROUGE Scores: {rouge_scores}")


        wandb.log({
            "test_loss": test_loss,
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL']
                 })

        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": eval_loss
        })

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}")

        if eval_loss < prev_loss:
            prev_loss = eval_loss
            torch.save(model.state_dict(), "fine_tuned_gpt_neo_summarization_small_data.pt")

#    model.load_state_dict(torch.load("fine_tuned_gpt_neo_summarization.pt", map_location=device))
    
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")

    
    rouge_scores = calculate_rouge(model, test_loader, tokenizer, MAX_LENGTH, device)
    print(f"ROUGE Scores: {rouge_scores}")

    
    wandb.log({
        "test_loss": test_loss,
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL']
    })

    wandb.finish()

if __name__ == "__main__":
    main()
