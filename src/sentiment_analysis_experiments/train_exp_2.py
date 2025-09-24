import emoji
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------
# Load data
# ------------------------------
train_df = pd.read_csv("data/sentiment/train.csv")
dev_df = pd.read_csv("data/sentiment/dev.csv")
test_df = pd.read_csv("data/sentiment/test.csv")

print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)
print("Test shape:", test_df.shape)
print("Train label counts:\n", train_df['label'].value_counts())
print("Dev label counts:\n", dev_df['label'].value_counts())

# ------------------------------
# Explore basics
# ------------------------------
for label in train_df['label'].unique():
    print(f"\nSample texts for label {label}:")
    print(train_df[train_df['label']==label]['text'].sample(5).tolist())

train_df['text_len'] = train_df['text'].apply(len)
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
print("Average text length:", train_df['text_len'].mean())
print("Average words per text:", train_df['word_count'].mean())

def count_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

train_df['emoji_count'] = train_df['text'].apply(count_emojis)
print("Total emojis in train:", train_df['emoji_count'].sum())

print("Missing values:\n", train_df.isnull().sum())

# ------------------------------
# Preprocessing function
# ------------------------------
# Load spacy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def clean_text(text, lowercase=True, remove_stop=False, emoji_to_text=True, fix_typos=False, lemmatize=False):
    if lowercase:
        text = text.lower()
    
    if fix_typos:
        text = re.sub(r"\bdont\b", "don't", text)
        text = re.sub(r"\bcant\b", "can't", text)
        text = re.sub(r"\bwont\b", "won't", text)
    
    # Handle emojis
    if emoji_to_text:
        text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove URLs and punctuation
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s:]', '', text)
    
    # Tokenization for stopwords / lemmatization
    doc = nlp(text)
    tokens = []
    for token in doc:
        t = token.lemma_ if lemmatize else token.text
        if remove_stop and t in stop_words:
            continue
        tokens.append(t)
    
    return " ".join(tokens)

# Apply preprocessing
train_df['clean_text'] = train_df['text'].apply(lambda x: clean_text(x, remove_stop=False, emoji_to_text=True))
dev_df['clean_text'] = dev_df['text'].apply(lambda x: clean_text(x, remove_stop=False, emoji_to_text=True))

# ------------------------------
# TF-IDF Vectorization
# ------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df['clean_text'])
y_train = train_df['label']

X_dev = vectorizer.transform(dev_df['clean_text'])
y_dev = dev_df['label']

# ------------------------------
# Train Logistic Regression
# ------------------------------
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Evaluate
# ------------------------------
y_pred = model.predict(X_dev)

print("Accuracy:", accuracy_score(y_dev, y_pred))
print("\nClassification Report:\n", classification_report(y_dev, y_pred, digits=4))

# ------------------------------
# Example clean text
# ------------------------------
print("Example cleaned text:\n", train_df["clean_text"].iloc[1])
