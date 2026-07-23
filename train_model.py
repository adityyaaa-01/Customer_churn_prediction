import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("churn.csv")

# Remove customerID
cleanDF = df.drop("customerID", axis=1)

label_encoders = {}

for col in cleanDF.columns:
    if cleanDF[col].dtype == object:
        le = LabelEncoder()
        cleanDF[col] = le.fit_transform(cleanDF[col])
        label_encoders[col] = le

X = cleanDF.drop("Churn", axis=1)
y = cleanDF["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(class_weight="balanced")

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save model and preprocessing objects
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoders.pkl")

# Save metrics
metrics = {
    "accuracy": accuracy,
    "report": report
}

joblib.dump(metrics, "metrics.pkl")

print("All files saved successfully.")