# === Import Library ===
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

# === Koneksi ke Database PostgreSQL ===
engine = create_engine("postgresql://postgres:admin@localhost:5432/hmsales_dw")

# === Load Data dari View v_sales_summary ===
df = pd.read_sql("SELECT * FROM v_sales_summary", engine)

# === Buat Label Klasifikasi Profit (1) atau Loss (0) ===
df["is_profit"] = (df["profit"] >= 0).astype(int)

# === Tentukan Fitur yang Digunakan dan Drop Missing Values ===
features = ["discount", "quantity", "sales", "category", "sub_category", "ship_mode", "region"]
df_model = df[features + ["is_profit"]].dropna()

# === One-hot Encoding untuk Fitur Kategorikal ===
df_encoded = pd.get_dummies(df_model, columns=["category", "sub_category", "ship_mode", "region"])

# === Pisahkan Fitur dan Label ===
X = df_encoded.drop("is_profit", axis=1)
y = df_encoded["is_profit"]

# === Split Data ke Train dan Test (80/20) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Normalisasi Fitur untuk KNN dan SVM ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Simpan Scaler ===
joblib.dump(scaler, "model/scaler.pkl")

# === Model 1: Random Forest ===
print("[1] RANDOM FOREST CLASSIFIER")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Training model
rf_pred = rf_model.predict(X_test)  # Prediksi
# Evaluasi model
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}\n")
# Simpan model + fitur yang digunakan
joblib.dump((rf_model, X.columns.tolist()), "model/rf_model.pkl")

# === Model 2: Decision Tree ===
print("[2] DECISION TREE CLASSIFIER")
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}\n")
joblib.dump((dt_model, X.columns.tolist()), "model/dt_model.pkl")

# === Model 3: K-Nearest Neighbors ===
print("[3] K-NEAREST NEIGHBORS CLASSIFIER")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)  # Menggunakan data yang sudah discaling
knn_pred = knn_model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, knn_pred):.4f}\n")
joblib.dump((knn_model, X.columns.tolist()), "model/knn_model.pkl")

# === Model 4: Support Vector Machine ===
print("[4] SUPPORT VECTOR MACHINE CLASSIFIER")
svm_model = SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}\n")
joblib.dump((svm_model, X.columns.tolist()), "model/svm_model.pkl")

# === Ringkasan Akurasi Semua Model ===
print("== MODEL SUMMARY ==")
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
