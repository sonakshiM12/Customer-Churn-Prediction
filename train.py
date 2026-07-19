import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Create model folder
# ==========================================
os.makedirs("model", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("data/churn.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================================
# Data Cleaning
# ==========================================
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ==========================================
# Encode Categorical Columns
# ==========================================
encoders = {}

for column in df.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

joblib.dump(encoders, "model/encoders.pkl")

# ==========================================
# Features & Target
# ==========================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

joblib.dump(list(X.columns), "model/features.pkl")

# ==========================================
# Split Data
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Train Model
# ==========================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Evaluate
# ==========================================
y_pred = model.predict(X_test)

print("\nModel Accuracy:")
print(f"{accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importance = (
    pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    )
    .sort_values(by="Importance", ascending=False)
)

print("\nFeature Importance:")
print(importance)

# ==========================================
# Save Model
# ==========================================
joblib.dump(model, "model/churn_model.pkl")

print("\nFiles Saved Successfully!")
print("✔ model/churn_model.pkl")
print("✔ model/encoders.pkl")
print("✔ model/features.pkl")
print("\nProject Training Completed Successfully!")
