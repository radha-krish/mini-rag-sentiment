# train_rf_embeddings.py

import pandas as pd
import emoji
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import spacy
from nltk.corpus import stopwords
import nltk

# ------------------------------
# Ensure NLTK stopwords are downloaded
# ------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# ------------------------------
# Load data
# ------------------------------
train_df = pd.read_csv("data/sentiment/train.csv")
dev_df   = pd.read_csv("data/sentiment/dev.csv")
test_df  = pd.read_csv("data/sentiment/test.csv")

print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)
print("Test shape:", test_df.shape)

# ------------------------------
# Preprocessing
# ------------------------------
nlp = spacy.load("en_core_web_sm")

def clean_text(text, lowercase=True, remove_stop=False, emoji_to_text=True, fix_typos=False, lemmatize=False):
    if lowercase:
        text = text.lower()
    if fix_typos:
        text = re.sub(r"\bdont\b", "don't", text)
        text = re.sub(r"\bcant\b", "can't", text)
        text = re.sub(r"\bwont\b", "won't", text)
    if emoji_to_text:
        text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s:]', '', text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        t = token.lemma_ if lemmatize else token.text
        if remove_stop and t in stop_words:
            continue
        tokens.append(t)
    return " ".join(tokens)

train_df['clean_text'] = train_df['text'].apply(lambda x: clean_text(x))
dev_df['clean_text']   = dev_df['text'].apply(lambda x: clean_text(x))

# ------------------------------
# Feature extraction using embeddings
# ------------------------------
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding training data...")
X_train = embedding_model.encode(train_df['clean_text'], convert_to_numpy=True)
y_train = train_df['label'].values

print("Encoding dev data...")
X_dev = embedding_model.encode(dev_df['clean_text'], convert_to_numpy=True)
y_dev = dev_df['label'].values

# ------------------------------
# Train Random Forest
# ------------------------------
rf = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42)
rf.fit(X_train, y_train)

# ------------------------------
# Evaluate
# ------------------------------
y_train_pred = rf.predict(X_train)
y_dev_pred   = rf.predict(X_dev)

print("\n=== Random Forest Accuracy ===")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Dev Accuracy:   {accuracy_score(y_dev, y_dev_pred):.4f}\n")

print("=== Classification Report (Dev Set) ===")
print(classification_report(y_dev, y_dev_pred, digits=4))

# ------------------------------
# Example cleaned text
# ------------------------------
print("Example cleaned text:\n", train_df['clean_text'].iloc[1])
