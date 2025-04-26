import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Loading model
GNB = joblib.load("Gaussian Naive Bayes")

# Reading Data
df = pd.read_excel("churn_dataset.xlsx")

# Streamlit App
st.title("Customer Churn Prediction App")

# Calculate min, max, and mean for sliders
age_min = int(df['Age'].min())
age_max = int(df['Age'].max())
age_mean = int(df['Age'].mean())

tenure_min = int(df['Tenure'].min())
tenure_max = int(df['Tenure'].max())
tenure_mean = int(df['Tenure'].mean())


# Input fields
age = st.slider("Age", age_min, age_max, age_mean)
tenure = st.slider("Tenure", tenure_min, tenure_max, tenure_mean)
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode Gender
gender_encoded = 1 if gender == "Male" else 0

# Prepare input for prediction
input_data = np.array([[age, tenure, gender_encoded]])

# Predict Churnes
prediction = GNB.predict(input_data)[0]
prediction_prob = GNB.predict_proba(input_data)[0]

# Labeling Targets
result_label = {0: "Not Churned", 1: "Churned"}

# Show Prediction
st.subheader("Prediction Results")
st.write(f"Predicted Churn: {result_label[prediction]}")

# Show Probabilities
st.subheader("Prediction Probability")
st.write(f"Churned: {prediction_prob[1]:.2%}")
st.write(f"Not Churned: {prediction_prob[0]:.2%}")