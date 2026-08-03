train_code = '''
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import time
import os
from config import Config

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_balanced_data(examples_per_class=200):
    print("Loading dataset...")
    train_data = fetch_20newsgroups(subset="train",
                                    remove=("headers","footers","quotes"))
    test_data  = fetch_20newsgroups(subset="test",
                                    remove=("headers","footers","quotes"))

    # balance training data
    balanced_texts  = []
    balanced_labels = []
    for class_idx in range(20):
        mask   = np.array(train_data.target) == class_idx
        texts  = np.array(train_data.data)[mask]
        labels = np.array(train_data.target)[mask]
        n      = min(examples_per_class, len(texts))
        balanced_texts.extend(texts[:n].tolist())
        balanced_labels.extend(labels[:n].tolist())

    print(f"Train: {len(balanced_texts)} | Test: {Config.TEST_SIZE}")
    return (balanced_texts, balanced_labels,
            test_data.data[:Config.TEST_SIZE],
            test_data.target[:Config.TEST_SIZE].tolist(),
            train_data.target_names)

def train():
    # load data
    train_texts, train_labels, test_texts, test_labels, categories = \\
        load_balanced_data(Config.EXAMPLES_PER_CLASS)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)

    # datasets
    train_dataset = NewsDataset(train_texts,  train_labels,
                                tokenizer, Config.MAX_LEN)
    test_dataset  = NewsDataset(test_texts,   test_labels,
                                tokenizer, Config.MAX_LEN)

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,
                              batch_size=Config.BATCH_SIZE, shuffle=False)

    # model
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=Config.NUM_CLASSES
    ).to(Config.DEVICE)

    # optimizer and scheduler
    optimizer    = AdamW(model.parameters(),
                         lr=Config.LEARNING_RATE,
                         weight_decay=Config.WEIGHT_DECAY)
    total_steps  = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    best_test_acc = 0
    best_epoch    = 0

    for epoch in range(Config.EPOCHS):
        start = time.time()

        # training
        model.train()
        train_correct = train_total = 0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels         = batch["label"].to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            predicted     = outputs.logits.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total   += labels.size(0)

        # evaluation
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(Config.DEVICE)
                attention_mask = batch["attention_mask"].to(Config.DEVICE)
                labels         = batch["label"].to(Config.DEVICE)
                outputs        = model(input_ids=input_ids,
                                      attention_mask=attention_mask)
                predicted      = outputs.logits.argmax(dim=1)
                test_correct  += (predicted == labels).sum().item()
                test_total    += labels.size(0)

        train_acc = train_correct/train_total * 100
        test_acc  = test_correct/test_total  * 100
        elapsed   = time.time() - start

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}% | "
              f"Time: {elapsed:.0f}s")

        # save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch    = epoch + 1
            os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(Config.MODEL_SAVE_PATH)
            tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
            print(f"  → Best model saved (epoch {best_epoch})")

    print(f"\\nTraining complete.")
    print(f"Best test accuracy: {best_test_acc:.1f}% at epoch {best_epoch}")

if __name__ == "__main__":
    train()
'''

with open('news_classifier_project/src/train.py', 'w') as f:
    f.write(train_code)
print("train.py written")