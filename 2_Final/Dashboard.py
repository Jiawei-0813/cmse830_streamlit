import os
import pandas as pd
import streamlit as st

# Set up the main structure
st.set_page_config(layout="wide", page_title="London Bike-Sharing Trends")

st.title("🚴 Exploring London Bike-Sharing Trends")

st.markdown("""
    ### Welcome to the Ride! 

    Bike-sharing is transforming how cities move, and London is no exception. It’s fast, affordable, and eco-friendly—perfect for navigating the city or soaking up its iconic sights. 

    More than just a way to get from point A to point B, London’s bike-sharing is a culture, a movement, and a story. From early-morning commutes to leisurely weekend rides through Hyde Park, these bikes connect people to the city in a way that’s both practical and personal.

    Every ride brings people closer—to each other, to landmarks, and to experiences that only two wheels can offer.
    """)

st.divider()

st.markdown("""
            **In August 2023, 776,527 of bike rides painted London’s streets with motion**. But what drives this rhythm? 
            
            This dashboard takes you on a journey to uncover how weather, time of day, and station popularity shaped London’s bike-sharing ecosystem. 
            """)

col1, col2 = st.columns([1, 1.3])

with col1:
    st.image(
    'http://gudphoto.com/bikenyc/wp-content/uploads/2012/04/20120418-DSCF0792.jpg',
    caption='Bike-sharing in London',
    use_container_width=True
)
with col2:
    st.markdown("""
                Here’s what you’ll explore:
            - **Weather and Rides**: Does a sunny day bring out more riders? How about rainy afternoons?
            - **Time of Day Trends**: When do Londoners ride the most—early mornings, busy evenings, or weekends?
            - **Station Hotspots**: Where do most rides start, and where do they end?
            """)

# --- Load Data ---
# Load bike-sharing data
@st.cache_data
def load_bike_data():
    path = os.path.join(os.getcwd(), 'data/0_LondonBikeJourneyAug2023_small.csv')
    bike_0 = pd.read_csv(path)
    return bike_0

@st.cache_data
def fetch_and_save_weather_data():
    try:
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
        weather_0.to_csv('data/0_london_weather_2023.csv', index=False)
        return weather_0
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        st.stop()

# Store in session_state
if "bike_data_raw" not in st.session_state:
    st.session_state["bike_data_raw"] = load_bike_data()
if "weather_data_raw" not in st.session_state:
    st.session_state["weather_data_raw"] = fetch_and_save_weather_data()
