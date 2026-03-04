from importlib.resources import Package

import streamlit as st # pyright: ignore[reportMissingImports]
import joblib # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]

# ==============================
# 1️ LOAD MODEL AND COLUMNS
# ==============================
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("columns.pkl")

# ==============================
# 2️ PAGE TITLE
# ==============================
st.title("🎬 Netflix Churn Prediction App")

st.write("Enter customer details below:")

# ==============================
# 3️ USER INPUT
# ==============================
age = st.number_input("Age", min_value=10, max_value=100, value=25)

watch_time = st.number_input("Watch Time (Hours per Week)", 
                             min_value=0.0, 
                             max_value=100.0, 
                             value=10.0)

country = st.selectbox("Country", 
                       ["Brazil", "Canada", "France", "Germany", "India"])

genre = st.selectbox("Favorite Genre", 
                     ["Action", "Comedy", "Horror", "Romance", "Sci-Fi"])

# ==============================
# 4️ PREDICT BUTTON
# ==============================
if st.button("Predict Churn"):

    # Create input dictionary
    input_dict = {
        "Age": age,
        "Watch_Time_Hours": watch_time,
        "Country": country,
        "Favorite_Genre": genre
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_df = pd.get_dummies(input_df)

    # Match training columns exactly
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)[0]

    # ==============================
    # 5️ SHOW RESULT
    # ==============================
    if prediction == 1:
        st.error(" This customer is likely to CHURN.")
    else:
        st.success(" This customer is likely to STAY.")
# DEBUG: Check package type and content

print(type(Package))
print(Package)