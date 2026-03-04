import streamlit as st
import joblib
import pandas as pd

# Load model
package = joblib.load("churn_model.pkl")
model = package["model"]
feature_columns = package["features"]

st.title("Customer Churn Prediction App")

# User inputs
age = st.number_input("Age", min_value=10, max_value=100)
watch_time = st.number_input("Watch Time Hours")

country = st.selectbox("Country", ["USA", "UK", "France"])
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
genre = st.selectbox("Favorite Genre", ["Drama", "Comedy", "Sci-Fi", "Documentary"])

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Watch_Time_Hours": [watch_time],
    "Country": [country],
    "Subscription_Type": [subscription],
    "Favorite_Genre": [genre]
})

# Convert to dummies
input_data = pd.get_dummies(input_data)

# Match training columns
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is NOT likely to churn")