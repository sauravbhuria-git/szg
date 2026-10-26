
# Experiment 6: NLP-Based SMS Spam Detection
# TF-IDF + Multinomial Naive Bayes
# Dataset: SMSSpamCollection.txt

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load Dataset
file_path = "SMSSpamCollection.txt"

data = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

data.dropna(inplace=True)
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].astype(str)

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("\nSample data:")
print(data.head())

# 2. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["message"] = data["message"].apply(clean_text)

# 3. Separate input and output
X = data["message"]
y = data["label"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# 5. Create and train NLP model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Evaluate model and display accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100

print("\n========== MODEL EVALUATION ==========")
print("Total messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
print(f"Model Accuracy: {accuracy_percent:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("======================================")

# 7. Save trained model
joblib.dump(model, "nlp_spam_model.pkl")
print("\nModel saved as nlp_spam_model.pkl")

# 8. Graph 1: Spam vs Ham distribution
plt.figure(figsize=(6, 5))
sns.countplot(data=data, x="label")
plt.title("SMS Spam vs Ham Distribution")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.savefig("graph1_message_distribution.png", dpi=300)
plt.show()

# 9. Graph 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("graph2_confusion_matrix.png", dpi=300)
plt.show()

# 10. Graph 3: Precision, Recall and F1-Score
report = classification_report(
    y_test,
    y_pred,
    labels=["ham", "spam"],
    output_dict=True,
    zero_division=0
)

metrics = pd.DataFrame({
    "Class": ["Ham", "Spam"],
    "Precision": [
        report["ham"]["precision"],
        report["spam"]["precision"]
    ],
    "Recall": [
        report["ham"]["recall"],
        report["spam"]["recall"]
    ],
    "F1-Score": [
        report["ham"]["f1-score"],
        report["spam"]["f1-score"]
    ]
})

metrics_plot = metrics.set_index("Class")
metrics_plot.plot(kind="bar", figsize=(8, 5))

plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graph3_model_performance.png", dpi=300)
plt.show()

# 11. Graph 4: Model Accuracy (95.96% if reproduced)
plt.figure(figsize=(6, 5))
bars = plt.bar(
    ["Naive Bayes Model"],
    [accuracy_percent],
    width=0.45
)

plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("NLP Model Accuracy")
plt.bar_label(bars, fmt="%.2f%%", padding=5)
plt.tight_layout()
plt.savefig("graph4_model_accuracy.png", dpi=300)
plt.show()

print(f"\nFinal Model Accuracy: {accuracy_percent:.2f}%")

# 12. Load saved model and test a new SMS
loaded_model = joblib.load("nlp_spam_model.pkl")

message = "Congratulations! You have won a free prize. Call now!"
prediction = loaded_model.predict([clean_text(message)])[0]

print("\n========== NEW SMS PREDICTION ==========")
print("New SMS:", message)
print("Predicted Class:", prediction.upper())
print("========================================")

print("\nAll 4 graphs saved successfully!")
