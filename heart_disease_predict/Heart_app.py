import streamlit as st
import pandas as pd
import joblib 

st.set_page_config(
    page_title="CardioCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load saved model, scaler, and expected columns

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load("columns.pkl")


st.title("❤️ CardioCare AI")
st.subheader("AI Powered Heart Disease Prediction System")
st.markdown("Provide the following details to check your heart stroke risk: ")

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#d63384;
    text-align:center;
}

.stButton>button{
    background:#dc3545;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:55px;
    width:100%;
}

.stButton>button:hover{
    background:#bb2d3b;
}

.result-box{
    padding:20px;
    border-radius:10px;
    background:white;
    box-shadow:0px 0px 10px gray;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.magnific.com/free-vector/tiny-cardiology-doctor-nurse-examining-heart-blood-pressure-prescribing-treatment-medical-cardiovascular-checkup-flat-vector-illustration-anatomy-hospital-heart-diseases-health-care-concept_74855-20963.jpg?semt=ais_test_b&w=740&q=80", width=120)

st.sidebar.title("CardioCare AI")

st.sidebar.info("""
This application predicts the possibility of heart disease using
Machine Learning.
""")

st.sidebar.success("Model : K-Nearest Neighbors")



# Collect the user input

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex",["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG",["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button('Predict'):

    # Create a raw input dictionary
    raw_input ={
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExcerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

# Create input dataframe

    input_df = pd.DataFrame([raw_input])

# Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

# Reorder columns
    input_df = input_df[expected_columns]

# Scale the input
    scaled_input = scaler.transform(input_df)

#Make Prediction
    prediction = model.predict(scaled_input)[0]

# Show result


    if prediction == 1:
        st.error("🔴 High Risk of Heart Disease")

        st.warning("""
    Please consult a cardiologist.
    Maintain a healthy diet and exercise regularly.
    """)
    else:
        st.success("Low Risk of Heart Disease")
    st.subheader("Patient Summary")

    st.dataframe(input_df)
    st.markdown("---")

st.markdown(
"""
<div class='footer'>

❤️ CardioCare AI <br>

Developed by <b>Uttam Kumar Singh</b><br>

Powered by Streamlit & Scikit-Learn

</div>
""",
unsafe_allow_html=True
)
    

