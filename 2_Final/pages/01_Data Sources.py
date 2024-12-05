import streamlit as st
import pandas as pd
from data_loader import load_bike_data, load_weather_data

st.set_page_config(
    page_title="Data Sources",
    layout="wide"
)

# Ensure data is loaded into session_state
if "bike_data_raw" not in st.session_state:
    try:
        st.session_state["bike_data_raw"] = load_bike_data()
    except FileNotFoundError as e:
        st.error(f"Error loading bike data: {e}")
        st.stop()

if "weather_data_raw" not in st.session_state:
    try:
        st.session_state["weather_data_raw"] = load_weather_data()
    except FileNotFoundError as e:
        st.error(f"Error loading weather data: {e}")
        st.stop()

# Debug paths
import os
st.write("Current Working Directory:", os.getcwd())
st.write("Does bike data exist?", os.path.exists("2_Final/data/0_LondonBikeJourneyAug2023_small.csv"))
st.write("Does weather data exist?", os.path.exists("2_Final/data/0_london_weather_2023.csv"))


st.title("Data Sources")

tab1, tab2 = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset"])

with tab1:
    st.subheader('🚴 London Bike-Sharing Dataset')
    st.write("""The bike dataset contains detailed records of individual bike trips from the Transport for London (TfL) bike-sharing system. """)
    
    st.write('**Data Access**: ')
    st.write('[Kaggle Dataset](https://www.kaggle.com/datasets/kalacheva/london-bike-share-usage-dataset)')  
       
    # Load the data
    bike_0 = st.session_state["bike_data_raw"]
    if st.checkbox("View raw bike data"):
        st.dataframe(bike_0.head())

    with st.expander("Check Raw Bike Data"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.write(bike_0.dtypes.to_frame('Type'))

        with col2: 
            col1, col2 = st.columns([1, 1])
            with col1:
                missing_values = bike_0.isnull().sum()
                if missing_values.sum() == 0:
                    st.markdown("<span style='color: #5F9EA0;'>No missing values found.</span>", unsafe_allow_html=True)
                else:
                    missing_percent = (missing_values / len(bike_0)) * 100
                    missing_summary = pd.DataFrame({
                    'Missing Values': missing_values,
                    '% Missing': missing_percent
                    })
                    st.markdown(missing_summary.style.applymap(lambda x: 'color: red' if x > 0 else 'color: green').render(), unsafe_allow_html=True)
            
            with col2:
                num_duplicates = bike_0.duplicated().sum()
                if num_duplicates > 0:
                    st.markdown(f"<div style='color: #5F9EA0;'>Number of duplicate rows: {num_duplicates}</div>", unsafe_allow_html=True)

                    bike_cleaned = bike_0.drop_duplicates()
                    st.markdown("<div style='color: #5F9EA0;'>- Duplicates have been removed.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #5F9EA0;'>No duplicates found.</div>", unsafe_allow_html=True)
                    num_duplicates = bike_0.duplicated().sum()
                    
            st.write("**🛠 Suggested Adjustments**")
            st.markdown("""
            - **`Start date` and `End date`**: Convert to `datetime` format
            - **`Total duration (ms)`**: Convert from milliseconds to minutes           
            - **`Total duration`**: Redundant and dropped
            - **`Bike model`**: Convert to `category` type 
            """)
            st.write("Also,")
            st.markdown("""
            - Create a **`date`** column in `yyyy-mm-dd HH:MM` based on `Start date` for merging
            - Check consistency between station names and station numbers
            """)

    with st.container():
        st.markdown("""
            **Note**: Due to limited storage and the LFS quota being 80% utilized previously, 
            a random 10% of the original data was used for this demonstration.
            """)
    
    st.write("""
            - **Temporal**: Covers trips from August 1 to August 31, 2023
            - **Spatial**: Includes bike stations across London
            - **Trips**: Records 776,527 individual rides (10% with 77,653 trips in the demo)
            """)
    st.write('**Features**: ')
    st.write("""
        - **`Number`** (Cardinal): Unique identifier for each trip; Trip ID.  
        - **`Start date`** (Nominal): Date and time in `yyyy-mm-dd hh:mm:ss` format.  
        - **`Start station number`** (Cardinal): Unique identifier for the start station.  
        - **`Start station`** (Nominal, Categorical): Name of the start station.  
        - **`End date`** (Nominal): Date and time in `yyyy-mm-dd hh:mm:ss` format.  
        - **`End station number`** (Cardinal): Unique identifier for the end station.  
        - **`End station`** (Nominal, Categorical): Name of the end station.  
        - **`Bike number`** (Cardinal): Unique identifier for the bike used.  
        - **`Bike model`** (Categorical): Type of bike.  
        - **`Total duration`** (Ratio): Duration of the trip in seconds.  
        - **`Total duration (ms)`** (Ratio): Duration of the trip in milliseconds.  
        """, unsafe_allow_html=True)
    
       
with tab2:
    st.subheader('🌤️ London Weather Dataset')
    st.write("""
            The weather dataset contains historical records of key weather conditions. Open-Meteo partners with national 
                weather services to deliver accurate and reliable weather data, selecting the most suitable models for each location. It 
                provides easy access to high-resolution weather information, making it useful for analyzing how weather impacts activities 
                like bike-sharing.
                """)
    st.write('**Data Access**:')    
    
    st.write('[Open-Meteo API Documentation](https://open-meteo.com/en/docs/historical-weather-api)') 

    weather_0 = st.session_state["weather_data_raw"]

    if st.checkbox("View raw weather data"):
        st.dataframe(weather_0.head())

    with st.expander("Check Raw Weather Data"):
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.write(weather_0.dtypes.apply(lambda x: x.name).to_frame('Type').style.set_table_styles(
                        [{'selector': 'th', 'props': [('text-align', 'left')]},
                        {'selector': 'td', 'props': [('text-align', 'left')]}]
                    ).set_table_attributes('style="width: auto;"'))

            with col2: 
                col1, col2 = st.columns([1, 1])
                with col1:
                    missing_values = weather_0.isnull().sum()
                    if missing_values.sum() == 0:
                        st.markdown("<span style='color: #5F9EA0;'>No missing values found.</span>", unsafe_allow_html=True)
                    else:
                        missing_percent = (missing_values / len(weather_0)) * 100
                        missing_summary = weather_0.DataFrame({
                            'Missing Values': missing_values,
                            '% Missing': missing_percent
                        })
                        st.markdown(missing_summary.style.applymap(lambda x: 'color: red' if x > 0 else 'color: green').render(), unsafe_allow_html=True)
           
                with col2:
                    num_duplicates = weather_0.duplicated().sum()
                    if num_duplicates > 0:
                        st.markdown(f"<div style='color: #5F9EA0;'>Number of duplicate rows: {num_duplicates}</div>", unsafe_allow_html=True)

                        weather_cleaned = weather_0.drop_duplicates()
                        st.markdown("<div style='color: #5F9EA0;'>- Duplicates have been removed.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color: #5F9EA0;'>No duplicates found.</div>", unsafe_allow_html=True)
                        num_duplicates = weather_0.duplicated().sum()
                            
                st.write("**🛠 Suggested Adjustments**")
                st.markdown("""
                - **`date`**: Remove timezone information
                - **`weather_code`**: Map to weather descriptions
                - **`Date`** was extracted from `date` in `yyyy-mm-dd HH:MM` format for merging
                """)

    st.write("""
        - **Temporal**: Hourly weather data from August 1 to August 31, 2023 
        - **Spatial**: Weather data for London, United Kingdom, at 51.5085° N, -0.1257° E
    """)
    st.write('**Features**: ')
    st.write("""
        - **`date` (Nominal)**: Date and time of the observation in 'yyyy-mm-dd hh:mm:ss' format.  
        - **`temperature_2m` (Interval)**: Air temperature at 2 meters above ground in degrees Celsius.  
        - **`relative_humidity_2m` (Ratio)**: Relative humidity at 2 meters above ground as a percentage.  
        - **`apparent_temperature` (Interval)**: Feels-like temperature in degrees Celsius, combining wind chill and humidity.  
        - **`wind_speed_10m` (Ratio)**: Wind speed at 10 meters above ground in m/s.  
        - **`wind_direction_10m` (Circular Numeric)**: Wind direction at 10 meters above ground in degrees.  
        - **`weather_code` (Nominal)**: Weather condition represented by a numeric code.  
            """, unsafe_allow_html=True)
    
    # Hide the weather code information initially
    with st.expander("Show Detailed Weather Code Info"):
        st.image('https://github.com/Leftium/weather-sense/assets/381217/6b346c7d-0f10-4976-bb51-e0a8678990b3', use_container_width=True)
        st.write("""
        - **Code Description**:
        - **0**: Clear sky
        - **1, 2, 3**: Mainly clear, partly cloudy, and overcast
        - **45, 48**: Fog and depositing rime fog
        - **51, 53, 55**: Drizzle: Light, moderate, and dense intensity
        - **56, 57**: Freezing Drizzle: Light and dense intensity
        - **61, 63, 65**: Rain: Slight, moderate and heavy intensity
        - **66, 67**: Freezing Rain: Light and heavy intensity
        - **71, 73, 75**: Snow fall: Slight, moderate, and heavy intensity
        - **77**: Snow grains
        - **80, 81, 82**: Rain showers: Slight, moderate, and violent
        - **85, 86**: Snow showers slight and heavy
        - **95***: Thunderstorm: Slight or moderate
        - **96, 99***: Thunderstorm with slight and heavy hail
        - (*) Thunderstorm forecast with hail is only available in Central Europe
        """, unsafe_allow_html=True)

    # Add custom CSS to change the sidebar button format
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] button {
        background-color: transparent;
        color: black;
        font-weight: bold;
        font-size: 14px;
        font-family: 'Brush Script MT', cursive;
        font-weight: bold;
        }
        div[data-testid="stSidebar"] button:hover {
        background-color: #90EE90;
        color: white;
        }
        div[data-testid="stSidebar"] button::before {
        content: '🤔';
        font-size: 24px; /* Larger emoji */
        margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
