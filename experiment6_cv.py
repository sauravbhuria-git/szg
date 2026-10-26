
# Experiment 6: Computer Vision - Handwritten Digit Recognition
# Feature Extraction: HOG
# Classification Algorithm: Support Vector Machine (SVM)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from skimage.feature import hog

# 1. Load Dataset
digits = load_digits()

X_images = digits.images
y = digits.target

print("Dataset loaded successfully!")
print("Total images:", len(X_images))
print("Image size:", X_images[0].shape)
print("Available digits:", np.unique(y))

# 2. Display sample handwritten digits
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_images[i], cmap="gray")
    ax.set_title(f"Digit: {y[i]}")
    ax.axis("off")

plt.suptitle("Sample Handwritten Digit Images")
plt.tight_layout()
plt.savefig("cv_graph1_sample_digits.png", dpi=300)
plt.show()

# 3. Extract HOG Features
def extract_hog_features(images):
    features = []

    for image in images:
        feature = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        features.append(feature)

    return np.array(features)

X = extract_hog_features(X_images)

print("\nHOG feature extraction completed!")
print("Feature dataset shape:", X.shape)

# 4. Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))

# 5. Train SVM Classifier
model = SVC(kernel="linear")

model.fit(X_train, y_train)

print("\nSVM model training completed!")

# 6. Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100

print("\n========== MODEL EVALUATION ==========")
print(f"Model Accuracy: {accuracy_percent:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("======================================")

# 7. Save Trained Model and HOG Settings
model_data = {
    "model": model,
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}

joblib.dump(model_data, "cv_digit_model.pkl")

print("\nModel saved as cv_digit_model.pkl")

# 8. Graph 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.title("SVM Digit Recognition - Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.tight_layout()
plt.savefig("cv_graph2_confusion_matrix.png", dpi=300)
plt.show()

# 9. Graph 3: Precision, Recall and F1-Score
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(10)),
    output_dict=True,
    zero_division=0
)

metrics = pd.DataFrame({
    "Digit": [str(i) for i in range(10)],
    "Precision": [report[str(i)]["precision"] for i in range(10)],
    "Recall": [report[str(i)]["recall"] for i in range(10)],
    "F1-Score": [report[str(i)]["f1-score"] for i in range(10)]
})

metrics.set_index("Digit").plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Digit Classification Performance")
plt.xlabel("Handwritten Digit")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("cv_graph3_performance.png", dpi=300)
plt.show()

# 10. Graph 4: Model Accuracy
plt.figure(figsize=(6, 5))

bars = plt.bar(
    ["SVM Classifier"],
    [accuracy_percent],
    width=0.45
)

plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Computer Vision Model Accuracy")
plt.bar_label(bars, fmt="%.2f%%", padding=5)
plt.tight_layout()
plt.savefig("cv_graph4_accuracy.png", dpi=300)
plt.show()

print(f"\nFinal Model Accuracy: {accuracy_percent:.2f}%")

# 11. Load Model and Predict a Digit
loaded_data = joblib.load("cv_digit_model.pkl")
loaded_model = loaded_data["model"]

# Select one test image and its HOG features
sample_index = 0
sample_features = X_test[sample_index].reshape(1, -1)
actual_digit = y_test[sample_index]

predicted_digit = loaded_model.predict(sample_features)[0]

print("\n========== DIGIT PREDICTION ==========")
print("Actual Digit:", actual_digit)
print("Predicted Digit:", predicted_digit)

if actual_digit == predicted_digit:
    print("Result: Correct Prediction")
else:
    print("Result: Incorrect Prediction")

print("======================================")
print("\nAll graphs saved successfully!")
