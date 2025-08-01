import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.markdown( # Add a background image and overlay
    """
    <style>
    .stApp {
        background-image: url("https://static1.straitstimes.com.sg/s3fs-public/articles/2017/01/24/41264232_-_23_01_2017_-_tmrain24.jpg?VersionId=kp3BoxC191W4_W6IpgOJOy3.74vR9ZG2");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); /* White overlay with transparency */
        z-index: 0; /* Ensure the overlay is behind the content */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
mdata = joblib.load('best_gradient_boosting_model.pkl') # Load the model and feature columns
model = mdata['gbr_model'] # Load the model
feature_columns = mdata['feature_columns'] # Load the feature columns
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


temperature = st.number_input('Temperature (°C)', min_value=20.0, max_value=50.0, step=0.1) # Input for temperature
if temperature > 35: # Check if the temperature is unusually high
    st.warning("This temperature is unusually high for Singapore!")
elif temperature < 25: # Check if the temperature is unusually low
    st.warning("This temperature is unusually low for Singapore!")
wind_direction = st.selectbox('Wind Direction', ['East', 'North', 'North-East', 'North-West', 'South-East',
                                                  'South-West', 'South', 'West']) # Input for wind direction
month = st.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                          'August', 'September', 'October', 'November', 'December']) # Input for month

windspeed = st.slider('Windspeed (m/s)', 0.0, 10.0, 3.0, step=0.1)
if windspeed > 8:
    st.warning("This windspeed is unusually high for Singapore!")
humidity = st.slider('Humidity (%)', 50, 100, 85, step=1)
input_data = {
    'windspeed': windspeed,
    'temperature': temperature,
    'humidity': humidity,
} # Create input data dictionary with all features

for direction in ['East', 'North', 'North-East', 'North-West', 'South-East', 'South-West', 'South', 'West']:
    input_data[f'wind_direction_{direction}'] = 1 if wind_direction == direction else 0 # Create one-hot encoding for wind direction

for month_name in ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']:
    input_data[f'month_{month_name}'] = 1 if month == month_name else 0 # Create one-hot encoding for month

input_df = pd.DataFrame([input_data], columns=all_features) # Create DataFrame from input data

input_df = input_df.reindex(columns= feature_columns, fill_value=0) # Ensure the input DataFrame has the same columns as the model
if st.button('Predict'):
    y_pred_log = model.predict(input_df)[0] # Predict the log of the average rainfall
    y_pred = np.exp(y_pred_log)  # Convert log prediction back to original scale
    if y_pred < 1:
        rain_class = "Dry"
    elif y_pred < 3:
        rain_class = "Light Rain"
    elif y_pred < 7:
        rain_class = "Moderate Rain"
    else:
        rain_class = "Heavy Rain"
        st.warning("Heavy rainfall predicted! Please take necessary precautions.")

    st.success(f'The predicted average rainfall is: {y_pred:.2f} mm'
               f' ({rain_class})') # Display the predicted average rainfall