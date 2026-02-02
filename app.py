
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Heart Disease ML App", layout="wide")
st.title("❤️ Heart Disease Prediction – ML Model Only")
st.image(
    "heart_failureimage.png",
    use_container_width=True
)
name = st.text_input("Enter your name:")
# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('best_Heartfailure_model.pkl')  

model = load_model()

# Use model feature names

# Assumes your pipeline includes StandardScaler + classifier
feature_columns = model.named_steps['scaler'].feature_names_in_  # scikit-learn >=1.0


# Prediction Form

st.subheader("❤️ Predict Heart Disease")
col1, col2 = st.columns(2)

# Example input for numeric / categorical features
age = col1.slider("Age", 20, 80, 45)
sex = col1.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = col1.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = col1.number_input("Resting Blood Pressure", 80, 200, 120)
chol = col1.number_input("Cholesterol", 100, 400, 200)
fbs = col1.selectbox("Fasting Blood Sugar > 120", [0, 1])
thalach = col2.number_input("Max Heart Rate", 60, 220, 150)
exang = col2.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = col2.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = col2.selectbox("Slope", [0, 1, 2])
ca = col2.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = col2.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=model.named_steps['scaler'].feature_names_in_)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Store in session_state so other widgets can access
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probability

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {probability:.2f})")

# -------------------------------
# Optional: Probability visualization
# -------------------------------
st.subheader("Prediction Probability")
if st.checkbox("Show probability bar"):
    # Only plot if probability exists
    if 'probability' in st.session_state:
        probability = st.session_state['probability']
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "High Risk"], [1-probability, probability], color=["green", "red"])
        ax.set_ylabel("Probability")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("⚠️ Click 'Predict' first to see probability chart")



