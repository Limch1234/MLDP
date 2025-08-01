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

url_backgrounds = {
    'Dry': 'https://thumbs.dreamstime.com/b/singapore-september-close-up-marina-bay-sands-wonderful-cityscape-sunny-day-shot-three-towers-ressort-against-135571833.jpg',
    'Light Rain': 'https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-1910724454.jpg?c=original',
    'Moderate Rain': 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/0ddc54f5-5910-407a-8384-a82c5ef44c24/dcv1iia-008c77fc-837e-488b-8c5d-165013a2a95d.jpg/v1/fill/w_1280,h_854,q_75,strp/light_rain_heavy__traffic_singapore_sunset_by_capturing_the_light_dcv1iia-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9ODU0IiwicGF0aCI6IlwvZlwvMGRkYzU0ZjUtNTkxMC00MDdhLTgzODQtYTgyYzVlZjQ0YzI0XC9kY3YxaWlhLTAwOGM3N2ZjLTgzN2UtNDg4Yi04YzVkLTE2NTAxM2EyYTk1ZC5qcGciLCJ3aWR0aCI6Ijw9MTI4MCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.hb4uom4x_5FZtami28_lJREWjIhzatms_kvC_4vkoqw',
    'Heavy Rain': 'https://finbarrfallon.com/wp-content/uploads/2021/06/Storm_FinbarrFallon2-1000x667.jpg'
}

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


    url_background = backgrounds[rain_class] # Get the background image URL based on the predicted rainfall class
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{url_background}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5); /* White overlay with transparency */
            z-index: -1; /* Ensure the overlay is behind the content */
        }}
    </style>
    """, unsafe_allow_html=True)
st.success(f'The predicted average rainfall is: {y_pred:.2f} mm ({rain_class})') # Display the predicted average rainfall
