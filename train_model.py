import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred := model.predict(X_test)))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
