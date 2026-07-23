import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction App")
st.write("Simple and User-Friendly Customer Churn Prediction using Machine Learning")

@st.cache_data
def load_data():
    return pd.read_csv("churn.csv")

df = load_data()

if st.checkbox("Show Dataset"):
    st.dataframe(df)

cleanDF = df.drop("customerID", axis=1)

label_encoders = {}
for col in cleanDF.columns:
    if cleanDF[col].dtype == object:
        le = LabelEncoder()
        cleanDF[col] = le.fit_transform(cleanDF[col])
        label_encoders[col] = le

X = cleanDF.drop("Churn", axis=1)
y = cleanDF["Churn"]
feature_columns = X.columns

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")
metrics = joblib.load("metrics.pkl")

st.subheader("📊 Model Performance")

st.write(f"✅ Accuracy: {metrics['accuracy']:.2f}")

if st.checkbox("Show Classification Report"):
    st.text(metrics["report"])

important_features = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport"
]

user_input = {}

for col in important_features:
    st.write(
        col,
        df[col].dtype,
        df[col].min(),
        df[col].max(),
        df[col].mean() if is_numeric_dtype(df[col]) else "Categorical"
    )

    if is_numeric_dtype(df[col]):
        user_input[col] = st.number_input(
            col,
            value=float(df[col].mean())
        )
    else:
        user_input[col] = st.selectbox(
            col,
            df[col].dropna().unique()
        )

for col in feature_columns:
    if col not in user_input:
        if is_numeric_dtype(df[col]):
            user_input[col] = df[col].mean()
        else:
            user_input[col] = df[col].mode()[0]

input_df = pd.DataFrame([user_input])[feature_columns]

for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Customer is likely to **CHURN**")
    else:
        st.success("Customer is likely to **STAY**")
