import pandas as pd
import os

# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "dataset/firewall_logs.csv"

df = pd.read_csv(file_path)

print("========== ORIGINAL DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# ==========================================
# 2. Check Missing Values
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 3. Remove Rows with Missing Values
# ==========================================

df = df.dropna().copy()

print("\n========== AFTER REMOVING MISSING VALUES ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==========================================
# 4. Check Action Classes
# ==========================================

print("\n========== ACTION DISTRIBUTION ==========")
print(df["Action"].value_counts())


# ==========================================
# 5. Convert Action to Lowercase
# ==========================================

df["Action"] = df["Action"].astype(str).str.lower().str.strip()


# ==========================================
# 6. Display Final Dataset Information
# ==========================================

print("\n========== FINAL DATASET ==========")

print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nFinal Shape:")
print(df.shape)


# ==========================================
# 7. Save Processed Dataset
# ==========================================

os.makedirs("results", exist_ok=True)

output_file = "results/processed_firewall_logs.csv"

df.to_csv(output_file, index=False)

print("\nProcessed dataset saved to:")
print(output_file)