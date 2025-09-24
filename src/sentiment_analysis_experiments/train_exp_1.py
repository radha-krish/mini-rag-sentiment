import emoji
import pandas as pd
import re

train_df = pd.read_csv("data/sentiment/train.csv")
dev_df = pd.read_csv("data/sentiment/dev.csv")
test_df = pd.read_csv("data/sentiment/test.csv")

print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)
print("Test shape:", test_df.shape)
print("Train label counts:\n", train_df['label'].value_counts())
print("Dev label counts:\n", dev_df['label'].value_counts())


for label in train_df['label'].unique():
    print(f"\nSample texts for label {label}:")
    print(train_df[train_df['label']==label]['text'].sample(5).tolist())



# Text length
train_df['text_len'] = train_df['text'].apply(len)
print("Average text length:", train_df['text_len'].mean())

# Unique words
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
print("Average words per text:", train_df['word_count'].mean())


def count_emojis(text):
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

train_df['emoji_count'] = train_df['text'].apply(count_emojis)
total_emojis = train_df['emoji_count'].sum()
print("Total emojis in train:", total_emojis)


# Missing values
print("Missing values:\n", train_df.isnull().sum())




def preprocess_with_emoji(text):
    # Lowercase
    text = text.lower()
    # Replace URLs
    text = re.sub(r'https?://\S+', '', text)
    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))  # adds spaces around emoji codes
    # Optional: remove extra punctuation
    text = re.sub(r'[^\w\s:]', '', text)
    return text

# Apply to train/dev
train_df['clean_text'] = train_df['text'].apply(preprocess_with_emoji)
dev_df['clean_text'] = dev_df['text'].apply(preprocess_with_emoji)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # unigrams + bigrams

X_train = vectorizer.fit_transform(train_df['clean_text'])
y_train = train_df['label']

X_dev = vectorizer.transform(dev_df['clean_text'])
y_dev = dev_df['label']


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_dev)

print("Accuracy:", accuracy_score(y_dev, y_pred))
print("\nClassification Report:\n", classification_report(y_dev, y_pred, digits=4))

print(train_df["clean_text"].iloc[1])

