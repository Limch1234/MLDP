import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_random_forest_model.pkl')

st.title("Weather Forecast Prediction")
st.write("This app predicts the weather forecast based on various features.")

temperature_high = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=25)
temperature_low = st.number_input("Temperature Low (°C)", min_value=-10, max_value=50, value=15)
humidity_high = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
humidity_low = st.number_input("Humidity Low (%)", min_value=0, max_value=100, value=40)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=150, value=20)


if st.button("Predict"):
    input_data = pd.DataFrame({
        'temperature_high': [temperature_high],
        'temperature_low': [temperature_low],
        'humidity_high': [humidity_high],
        'humidity_low': [humidity_low],
        'wind_speed': [wind_speed]
    })
    
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    max_probability = probabilities.max()
    predicted_class = model.classes_[prediction[0]]
    st.write(f"{predicted_class} with a probability of {max_probability:.2f}")
