import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb

# Load the trained LightGBM model
model = lgb.Booster(model_file="models/lightgbm_model.lgb")

# Feature names expected by the model
features = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex',
    'BMI', 'GenHlth', 'Age', 'Education', 'Income',
    'MentHlth_Bin', 'PhysHlth_Bin'
]

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🔍 Diabetes Risk Prediction")
st.markdown("Fill the form below to get your diabetes risk prediction:")

# Input fields
user_input = {}
user_input['HighBP'] = st.radio("Do you have high blood pressure?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['HighChol'] = st.radio("Do you have high cholesterol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['CholCheck'] = st.radio("Have you checked cholesterol in last 5 years?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['Smoker'] = st.radio("Do you smoke?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['Stroke'] = st.radio("Have you had a stroke?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['HeartDiseaseorAttack'] = st.radio("Heart disease or heart attack history?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['PhysActivity'] = st.radio("Do you do physical activity?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['Fruits'] = st.radio("Do you consume fruits?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['Veggies'] = st.radio("Do you consume vegetables?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['HvyAlcoholConsump'] = st.radio("Do you consume heavy alcohol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['AnyHealthcare'] = st.radio("Do you have healthcare coverage?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['NoDocbcCost'] = st.radio("Can you afford seeing a doctor?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['DiffWalk'] = st.radio("Do you have difficulties walking?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
user_input['Sex'] = st.radio("Sex", ["Male", "Female"])
user_input['Sex'] = 1 if user_input['Sex'] == "Female" else 0
user_input['BMI'] = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=120.0, step=1.0)
user_input['GenHlth'] = st.radio("On a scale of 1 to 5, how would you rate your general health?", 
                    [1, 2, 3, 4, 5], 
                    format_func=lambda x: [
                        "Excellent", "Very good", "Good", "Fair", "Poor"
                    ][x - 1])
user_input['Age'] = st.number_input("Age", min_value=18.0, max_value=120.0, step=1.0)
user_input['Education'] = st.radio("What is your highest education level?",
                    [1, 2, 3, 4, 5, 6],
                    format_func=lambda x: [
                        "No education", "Elementary", "Some high school",
                        "High school graduate", "Some college/tech school", "College graduate"
                    ][x - 1])
user_input['Income'] = st.radio("Income level",
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    format_func=lambda x: [
                        "< $10k", "$10k-$15k", "$15k-$20k", "$20k-$25k",
                        "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"
                    ][x - 1])
user_input['MentHlth_Bin'] = st.radio("Mental health problems in last 30 days", 
                    [0, 1, 2, 3, 4], 
                    format_func=lambda x: [
                        "0 days", "1-5 days", "6-15 days", "16-29 days", "30 days"
                    ][x - 1])
user_input['PhysHlth_Bin'] = st.radio("Physical health problems in last 30 days", 
                    [0, 1, 2, 3, 4], 
                    format_func=lambda x: [
                        "0 days", "1-5 days", "6-15 days", "16-29 days", "30 days"
                    ][x - 1])

if st.button("Predict Diabetes Risk"):
    input_data = np.array([[float(user_input[f]) for f in features]])
    prob = model.predict(input_data)[0]
    risk = "🔴 High Risk" if prob >= 0.8 else "🟡 Moderate Risk" if prob >= 0.5 else "🟢 Low Risk"
    st.markdown(f"### {risk}")
    st.write(f"**Probability of diabetes:** {round(prob * 100, 2)}%")