
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. LOAD DATASET
df = pd.read_csv("data/sentiment_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# 2. CLEAN TWEET TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)

# Remove tweets that become empty after cleaning
df = df[df["clean_text"].str.len() > 0].copy()

print("\nText cleaning completed!")


# 3. PREPARE FEATURES AND LABELS
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. CONVERT TEXT INTO NUMERICAL FEATURES
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")


# 5. TRAIN THE MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# 6. EVALUATE THE MODEL
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 7. CREATE OUTPUT FOLDER
os.makedirs("outputs", exist_ok=True)


# 8. CONFUSION MATRIX
labels = ["negative", "positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Sentiment Analysis - Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()


# 9. SAVE MODEL ACCURACY CHART
plt.figure(figsize=(6, 4))
plt.bar(["Logistic Regression"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Sentiment Analysis Model Accuracy")
plt.tight_layout()
plt.savefig("outputs/model_accuracy.png", dpi=300)
plt.close()


# 10. TEST WITH SAMPLE TWEETS
sample_tweets = [
    "I love this product, it is amazing!",
    "This is the worst experience ever.",
    "I am happy with the service."
]

print("\nSample Predictions:")
for tweet in sample_tweets:
    cleaned = clean_text(tweet)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    print(f"Tweet: {tweet}")
    print(f"Predicted sentiment: {prediction}\n")


print("Charts saved in the outputs folder.")
print("Sentiment analysis completed successfully!")
