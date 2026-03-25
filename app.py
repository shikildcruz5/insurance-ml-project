import streamlit as st
import joblib
import pandas as pd

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# TITLE
# =========================
st.title("Insurance Customer Response Prediction")

# =========================
# USER INPUTS (FRIENDLY UI)
# =========================

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

# Age
age = st.number_input("Age", min_value=18, max_value=100)

# Region Code
region_code = st.number_input("Region Code", min_value=0)

# Driving License
driving_license = st.selectbox("Driving License", ["No", "Yes"])
driving_license = 0 if driving_license == "No" else 1

# Previously Insured
previously_insured = st.selectbox("Previously Insured", ["No", "Yes"])
previously_insured = 0 if previously_insured == "No" else 1

# Vehicle Age
vehicle_age = st.selectbox(
    "Vehicle Age",
    ["Less than 1 Year", "1-2 Years", "More than 2 Years"]
)

# Encode vehicle age
if vehicle_age == "Less than 1 Year":
    vehicle_age = 0
elif vehicle_age == "1-2 Years":
    vehicle_age = 1
else:
    vehicle_age = 2

# Vehicle Damage
vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"])
vehicle_damage = 0 if vehicle_damage == "No" else 1

# Annual Premium
annual_premium = st.number_input("Annual Premium")

# Policy Sales Channel
policy_channel = st.number_input("Policy Sales Channel")

# Vintage
vintage = st.number_input("Vintage")

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    # Create DataFrame (same format as training)
    data = pd.DataFrame([[gender, age, driving_license, region_code,
                          previously_insured, vehicle_age, vehicle_damage,
                          annual_premium, policy_channel, vintage]],
                        columns=[
                            "Gender", "Age", "Driving_License", "Region_Code",
                            "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
                            "Annual_Premium", "Policy_Sales_Channel", "Vintage"
                        ])

    try:
        # Prediction
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]

        # Output
        if prediction[0] == 1:
            st.success("Customer will RESPOND ✅")
        else:
            st.error("Customer will NOT respond ❌")

        # Probability
        st.write(f"Probability of Response: {round(probability * 100, 2)}%")

    except Exception as e:
        st.error(f"Error: {e}")