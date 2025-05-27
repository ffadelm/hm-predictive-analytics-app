import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Koneksi database
engine = create_engine("postgresql://postgres:admin@localhost:5432/hmsales_dw")
df = pd.read_sql("SELECT * FROM v_sales_summary", engine)

# Label: profit >= 0 → is_profit
df["is_profit"] = (df["profit"] >= 0).astype(int)

# Fitur yang digunakan
features = ["discount", "quantity", "sales", "category", "sub_category", "ship_mode", "region"]
df_model = df[features + ["is_profit"]].dropna()
df_encoded = pd.get_dummies(df_model, columns=["category", "sub_category", "ship_mode", "region"])

# Split data
X = df_encoded.drop("is_profit", axis=1)
y = df_encoded["is_profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === SCALING for KNN & SVM ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# === MODEL 1: RANDOM FOREST ===
print("[1] RANDOM FOREST CLASSIFIER")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}\n")
joblib.dump(rf_model, "rf_model.pkl")

# === MODEL 2: DECISION TREE ===
print("[2] DECISION TREE CLASSIFIER")
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}\n")
joblib.dump(dt_model, "dt_model.pkl")

# === MODEL 3: K-NEAREST NEIGHBORS (with scaled input) ===
print("[3] K-NEAREST NEIGHBORS CLASSIFIER")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, knn_pred):.4f}\n")
joblib.dump(knn_model, "knn_model.pkl")

# === MODEL 4: SUPPORT VECTOR MACHINE (with scaling & class_weight) ===
print("[4] SUPPORT VECTOR MACHINE CLASSIFIER")
svm_model = SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}\n")
joblib.dump(svm_model, "svm_model.pkl")

print("== MODEL SUMMARY ==")
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))