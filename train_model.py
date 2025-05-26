import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Koneksi database
engine = create_engine("postgresql://postgres:admin@localhost:5432/hmsales_dw")

# Load data dari view
df = pd.read_sql("SELECT * FROM v_sales_summary", engine)

# Label: profit >= 0 (1 = untung, 0 = rugi)
df["is_profit"] = (df["profit"] >= 0).astype(int)

# Fitur yang digunakan
features = [
    "discount", "quantity", "sales", "category", "sub_category", 
    "ship_mode", "region"
]
df_model = df[features + ["is_profit"]].dropna()

# One-hot encoding
df_encoded = pd.get_dummies(df_model, columns=["category", "sub_category", "ship_mode", "region"])

# Split data
X = df_encoded.drop("is_profit", axis=1)
y = df_encoded["is_profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== MODEL 1: RANDOM FOREST =====
print("[1] RANDOM FOREST CLASSIFIER")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

joblib.dump(rf_model, "rf_model.pkl")
print("Model saved as rf_model.pkl\n")

# ===== MODEL 2: DECISION TREE =====
print("[2] DECISION TREE CLASSIFIER")
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred))
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}")

joblib.dump(dt_model, "dt_model.pkl")
print("Model saved as dt_model.pkl")
