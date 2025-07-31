import streamlit as st
import numpy as np
import pandas as pd
import joblib

mdata = joblib.load('best_gradient_boosting_model.pkl')
model = mdata['gbr_model']
feature_columns = mdata['feature_columns']
# Define the features used in the model
all_features = ['windspeed', 'temperature', 'humidity', 'wind_direction_East', 'wind_direction_North',
                'wind_direction_North-East', 'wind_direction_North-West', 'wind_direction_South-East',
                'wind_direction_South-West', 'month_April', 'month_August', 'month_December',
                'month_February', 'month_January', 'month_July', 'month_June', 'month_March',
                'month_May', 'month_November', 'month_October', 'month_September',
                'wind_direction_South', 'wind_direction_West']

# Initialize an empty DataFrame with the required columns

st.title('Average Monthly Rainfall Prediction in Upp Changi Rd North')

st.markdown("Enter the weather data below to predict the average monthly rainfall:")


temperature = st.number_input('Temperature (°C)', min_value=20.0, max_value=50.0, step=0.1)
windspeed = st.slider('Windspeed (m/s)', 0.0, 10.0, 3.0, step=0.1)
humidity = st.slider('Humidity (%)', 50, 100, 85, step=1)
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

input_df = input_df.reindex(columns= feature_columns, fill_value=0)
if st.button('Predict'):
    y_pred_log = model.predict(input_df)[0]
    y_pred = np.exp(y_pred_log)  # Convert log prediction back to original scale
    st.success(f'The predicted average rainfall is: {y_pred:.2f} mm')