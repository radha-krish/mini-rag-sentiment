import emoji
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# ------------------------------
# Preprocessing
# ------------------------------
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

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
# Train multiple models
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_dev, y_pred, digits=4))

# ------------------------------
# Compare accuracies
# ------------------------------
print("\n=== Model Comparison ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
