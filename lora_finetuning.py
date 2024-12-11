from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import wandb
from rouge_score import rouge_scorer
import argparse
from peft import get_peft_model, LoraConfig

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                # Rank for LoRA
    lora_alpha=32,      # Alpha for LoRA scaling
    lora_dropout=0.1,   # Dropout for LoRA
    task_type='CAUSAL_LM'
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Ensure all parameters are trainable
for param in model.parameters():
    param.requires_grad = True

class CNNDailyMailDataset(Dataset):
    def __init__(self, split, tokenizer, max_length):
        self.data = pd.read_csv(f"./cnn_dailymail/{split}.csv")
        self.tokenizer = tokenizer
        self.max_length = max_length        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side='left'

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

    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(eval_loader)

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

            generated_ids = model.generate(input_ids=input_ids[:, :summary_length - 1], attention_mask=attention_mask[:, :summary_length - 1], max_length=summary_length)
            generated_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for gen, ref in zip(generated_summaries, reference_summaries):
                scores = scorer.score(ref, gen)
                for key in rouge_scores:
                    rouge_scores[key] += scores[key].fmeasure
                count += 1

    for key in rouge_scores:
        rouge_scores[key] /= count

    return rouge_scores

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = args.max_length

    train_dataset = CNNDailyMailDataset("train", tokenizer, MAX_LENGTH)
    val_dataset = CNNDailyMailDataset("validation", tokenizer, MAX_LENGTH)
    test_dataset = CNNDailyMailDataset("test", tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer, device)
        eval_loss = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}")

    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")

    rouge_scores = calculate_rouge(model, test_loader, tokenizer, MAX_LENGTH, device)
    print(f"ROUGE Scores: {rouge_scores}")

    wandb.finish()

if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    Parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    Parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    Parser.add_argument("--max_length", type=int, default=512, help="Maximum length")
    args = Parser.parse_args()
    main(args)
