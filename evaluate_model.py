import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "results/processed_firewall_logs.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.drop("Action", axis=1)
y = df["Action"]


# ==========================================
# 3. Load Saved Model and Encoder
# ==========================================

model = joblib.load("models/firewall_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# ==========================================
# 4. Encode Target
# ==========================================

y_encoded = label_encoder.transform(y)


# ==========================================
# 5. Create Same Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ==========================================
# 6. Make Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. Calculate Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ==========================================
# 8. Display Results
# ==========================================

print("\n==========================================")
print("           MODEL EVALUATION")
print("==========================================")

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


# ==========================================
# 9. Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# 10. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)


# ==========================================
# 11. Create Confusion Matrix Plot
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Action")
plt.ylabel("Actual Action")
plt.title("Firewall Action Classification - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved to:")
print("results/confusion_matrix.png")