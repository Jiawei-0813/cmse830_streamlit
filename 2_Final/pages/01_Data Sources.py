import streamlit as st

st.set_page_config(
    page_title="Data Sources",
    layout="wide"
)

st.title("Data Sources")

tab1, tab2 = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset"])

with tab1:
    st.subheader('🚴 London Bike-Sharing Dataset')
    st.write("""The bike dataset contains detailed records of individual bike trips from the Transport for London (TfL) bike-sharing system. """)
    st.write('**Data Access**: ')
    st.write('[Kaggle Dataset](https://www.kaggle.com/datasets/kalacheva/london-bike-share-usage-dataset)')  
    st.markdown("""
            Due to limited storage and the LFS quota being 80% utilized previously, 
            a random 10% of the original data was used for this demostration.
            """)
    st.write("""
            - **Temporal**: Covers trips from August 1 to August 31, 2023
            - **Spatial**: Includes bike stations across London
            - **Trips**: Records 776,527 individual rides
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
    if st.button("Show Detailed Weather Code Info", key="weather_code_button"):
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