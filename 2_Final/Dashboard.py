import streamlit as st
import pandas as pd

# Set up the main structure
st.set_page_config(layout="wide", page_title="London Bike-Sharing Trends")

st.title('London Bike-Sharing Trends')

st.subheader('Background')
st.write("""
            Bike-sharing services have become increasingly popular in urban areas as an affordable, convenient 
            and eco-friendly mode of transportation. In London, bike-sharing options are widely available, making 
            it easy for residents and visitors to explore the city on two wheels.
            """)

st.subheader('Goal')
st.write("""
            This project examines bike-sharing patterns in London, focusing on how weather conditions impact usage. 
            It aims to visualize the influence of factors like weather and time of day on bike-sharing trends.
            """)

st.image('http://gudphoto.com/bikenyc/wp-content/uploads/2012/04/20120418-DSCF0792.jpg', 
            caption='Bike-sharing in London',
            width=800
            )

# --- Load Data ---
# Load bike-sharing data
@st.cache_data
def load_bike_data():
    bike_0 = pd.read_csv('2_Final/data/0_london_weather_2023.csv')
    return bike_0

@st.cache_data
def fetch_and_save_weather_data():
    # Install
    # pip install openmeteo-requests
    # pip install requests-cache retry-requests numpy pandas

    # import requests
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 51.5085,
        "longitude": -0.1257,
        "start_date": "2023-08-01",
        "end_date": "2023-08-31",
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "weather_code", "wind_speed_10m", "wind_direction_10m"],
        "timezone": "Europe/London"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
 
    weather_0 = pd.DataFrame(data = hourly_data)

    # Save as a CSV file
    weather_0.to_csv('2_Final/data/0_london_weather_2023.csv', index=False)
    return weather_0

# Load datasets and store in session_state
st.session_state["bike_data_raw"] = load_bike_data()
st.session_state["weather_data_raw"] = fetch_and_save_weather_data()