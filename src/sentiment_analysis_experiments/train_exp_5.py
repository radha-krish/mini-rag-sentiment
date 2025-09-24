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
vectorizer =vectorizer = TfidfVectorizer(
    max_features=1000, 
    ngram_range=(1,2),
    min_df=2,      # ignore words appearing in <2 documents
    max_df=0.9     # ignore words appearing in >90% of documents
)

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


from sklearn.model_selection import GridSearchCV

# Random Forest hyperparameters to tune
rf_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)

print("\nBest RF params:", rf_grid.best_params_)
print("Best RF CV score:", rf_grid.best_score_)

# Evaluate on dev set
y_pred_rf = rf_grid.predict(X_dev)
print("RF tuned accuracy:", accuracy_score(y_dev, y_pred_rf))
print(classification_report(y_dev, y_pred_rf, digits=4))



# SVM hyperparameters to tune
svm_params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "loss": ["hinge", "squared_hinge"]
}

svm = LinearSVC(random_state=42, max_iter=5000)
svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1, verbose=2)
svm_grid.fit(X_train, y_train)

print("\nBest SVM params:", svm_grid.best_params_)
print("Best SVM CV score:", svm_grid.best_score_)

# Evaluate on dev set
y_pred_svm = svm_grid.predict(X_dev)
print("SVM tuned accuracy:", accuracy_score(y_dev, y_pred_svm))
print(classification_report(y_dev, y_pred_svm, digits=4))
print("\n=== Tuned Model Comparison ===")
print("Random Forest (tuned):", accuracy_score(y_dev, y_pred_rf))
print("SVM (tuned):", accuracy_score(y_dev, y_pred_svm))

# ------------------------------
# Compare accuracies (baseline)
# ------------------------------
print("\n=== Baseline Model Comparison ===")
for name, clf in models.items():
    y_train_pred = clf.predict(X_train)
    y_dev_pred = clf.predict(X_dev)
    print(f"{name}:")
    print(f"  Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Dev Accuracy:   {accuracy_score(y_dev, y_dev_pred):.4f}")
    print(classification_report(y_dev, y_dev_pred, digits=4))

# ------------------------------
# Tuned Random Forest
# ------------------------------
y_train_pred_rf = rf_grid.predict(X_train)
y_dev_pred_rf = rf_grid.predict(X_dev)
print("\nRandom Forest (Tuned):")
print(f"  Train Accuracy: {accuracy_score(y_train, y_train_pred_rf):.4f}")
print(f"  Dev Accuracy:   {accuracy_score(y_dev, y_dev_pred_rf):.4f}")
print(classification_report(y_dev, y_dev_pred_rf, digits=4))

# ------------------------------
# Tuned SVM
# ------------------------------
y_train_pred_svm = svm_grid.predict(X_train)
y_dev_pred_svm = svm_grid.predict(X_dev)
print("\nLinear SVM (Tuned):")
print(f"  Train Accuracy: {accuracy_score(y_train, y_train_pred_svm):.4f}")
print(f"  Dev Accuracy:   {accuracy_score(y_dev, y_dev_pred_svm):.4f}")
print(classification_report(y_dev, y_dev_pred_svm, digits=4))

# ------------------------------
# Summary
# ------------------------------
print("\n=== Tuned Model Accuracy Summary ===")
print(f"Random Forest (Tuned) - Train: {accuracy_score(y_train, y_train_pred_rf):.4f}, Dev: {accuracy_score(y_dev, y_dev_pred_rf):.4f}")
print(f"SVM (Tuned)           - Train: {accuracy_score(y_train, y_train_pred_svm):.4f}, Dev: {accuracy_score(y_dev, y_dev_pred_svm):.4f}")

