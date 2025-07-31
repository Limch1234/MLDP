import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('best_gradient_boosting_model.pkl')

all_features = ['windspeed', 'temperature', 'humidity', 'wind_direction_East', 'wind_direction_North',
                'wind_direction_North-East', 'wind_direction_North-West', 'wind_direction_South-East',
                'wind_direction_South-West', 'month_April', 'month_August', 'month_December',
                'month_February', 'month_January', 'month_July', 'month_June', 'month_March',
                'month_May', 'month_November', 'month_October', 'month_September',
                'wind_direction_South', 'wind_direction_West']

# Initialize an empty DataFrame with the required columns

st.title('Average Rainfall Prediction')

st.markdown("Enter the weather data below to predict the average rainfall:")

windspeed = st.number_input('Windspeed (m/s)', min_value=0.0, step=0.1)
temperature = st.number_input('Temperature (°C)', min_value=-30.0, max_value=50.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, step=1)
wind_direction = st.selectbox('Wind Direction', ['East', 'North', 'North-East', 'North-West', 'South-East',
                                                  'South-West', 'South', 'West'])
month = st.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                          'August', 'September', 'October', 'November', 'December'])
input_data = {
    'windspeed': windspeed,
    'temperature': temperature,
    'humidity': humidity,
}

for direction in ['East', 'North', 'North-East', 'North-West', 'South-East', 'South-West', 'South', 'West']:
    input_data[f'wind_direction_{direction}'] = 1 if wind_direction == direction else 0

for month_name in ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']:
    input_data[f'month_{month_name}'] = 1 if month == month_name else 0

input_df = pd.DataFrame([input_data], columns=all_features)

if st.button('Predict'):
    y_pred_log = model.predict(input_df)[0]
    y_pred = np.exp(y_pred_log)  # Convert log prediction back to original scale
    st.success(f'The predicted average rainfall is: {y_pred:.2f} mm')