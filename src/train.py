# train_transformer.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
import re
import emoji
import nltk
from nltk.corpus import stopwords

# ------------------------------
# Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

# ------------------------------
# Data Preprocessing
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s:]', '', text)
    return text

train_df = pd.read_csv("data/sentiment/train.csv")
dev_df   = pd.read_csv("data/sentiment/dev.csv")
test_df  = pd.read_csv("data/sentiment/test.csv")

train_df['clean_text'] = train_df['text'].apply(clean_text)
dev_df['clean_text']   = dev_df['text'].apply(clean_text)
test_df['clean_text']  = test_df['text'].apply(clean_text)

# ------------------------------
# Dataset class
# ------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ------------------------------
# Tokenizer and DataLoader
# ------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_dataset = SentimentDataset(train_df['clean_text'].tolist(),
                                 train_df['label'].tolist(),
                                 tokenizer,
                                 MAX_LEN)
dev_dataset = SentimentDataset(dev_df['clean_text'].tolist(),
                               dev_df['label'].tolist(),
                               tokenizer,
                               MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# Model
# ------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# ------------------------------
# Training loop
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # ------------------------------
    # Evaluate on training set
    # ------------------------------
    model.eval()
    train_preds, train_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(batch_preds)
            train_labels.extend(labels.cpu().numpy())
    train_acc = accuracy_score(train_labels, train_preds)
    print(f"Train Accuracy: {train_acc:.4f}")
    print("Train Classification Report:\n", classification_report(train_labels, train_preds, digits=4))
    
    # ------------------------------
    # Evaluate on dev set
    # ------------------------------
    dev_preds, dev_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            dev_preds.extend(batch_preds)
            dev_labels.extend(labels.cpu().numpy())
    dev_acc = accuracy_score(dev_labels, dev_preds)
    print(f"Dev Accuracy: {dev_acc:.4f}")
    print("Dev Classification Report:\n", classification_report(dev_labels, dev_preds, digits=4))

# ------------------------------
# Predict test set
# ------------------------------
test_dataset = SentimentDataset(test_df['clean_text'].tolist(), tokenizer=tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        test_preds.extend(batch_preds)

submission = pd.DataFrame({
    "text": test_df['text'],
    "label": test_preds
})
submission.to_csv("submissions/sentiment_test_predictions.csv", index=False)
print("\nSaved predictions to submissions/sentiment_test_predictions.csv")
