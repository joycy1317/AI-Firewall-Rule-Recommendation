import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Load Processed Dataset
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
# 3. Encode Target Classes
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget classes:")
for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)


# ==========================================
# 4. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. Create Models
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 6. Train and Compare Models
# ==========================================

results = {}

print("\n========== MODEL TRAINING ==========\n")

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print()


# ==========================================
# 7. Display Comparison
# ==========================================

print("========== MODEL COMPARISON ==========")

for name, accuracy in results.items():

    print(
        f"{name}: {accuracy * 100:.2f}%"
    )


# ==========================================
# 8. Select Best Model
# ==========================================

best_model_name = max(
    results,
    key=results.get
)

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print(
    "Best Accuracy:",
    f"{results[best_model_name] * 100:.2f}%"
)


# ==========================================
# 9. Save Best Model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/firewall_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print("\nBest model saved to:")
print("models/firewall_model.pkl")

print("\nLabel encoder saved to:")
print("models/label_encoder.pkl")