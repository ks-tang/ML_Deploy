import streamlit as st
import joblib

# Load du model
loaded_model = joblib.load('linear_regression_model.pkl')

# Load du scaler
loaded_scaler = joblib.load('scaler.pkl')


# Streamlit App
st.title("Student Test Score Predictor")
st.write("Enter the number of hours studied to predict the test score")

# User input
hours = st.number_input("Hours studied:", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        data = [[hours]]
        scaled_data = loaded_scaler.transform(data)
        prediction = loaded_model.predict(scaled_data)
        st.write(f"Predicted Test Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

        