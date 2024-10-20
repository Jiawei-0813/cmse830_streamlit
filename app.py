#!/usr/bin/env python

# Set up the Streamlit app
import streamlit as st
st.set_page_config(layout="wide") # Set page layout to wide

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # having issues with this module
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

st.title('London Bike-Sharing Trends')
# st.subheader('CMSE 830 Midterm Project')

# --- Load Data ---
# Load bike-sharing data
@st.cache_data
def load_data():
    bike_0 = pd.read_csv('datasets/1_LondonBikeJourneyAug2023.csv')
    return bike_0

bike_0 = load_data()

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
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "weather_code", "wind_speed_10m", "wind_direction_10m"],
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
    weather_0.to_csv('datasets/2_london_weather_2023.csv', index=False)
    return weather_0

weather_0 = fetch_and_save_weather_data()
    
# --- Sidebar ---
# Display menu
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go To', 
                        ['Overview',
                         'Data Sources', 
                         'Data Exploration', 
                         'Data Visualization'])

# Fun Fact
total_trips = bike_0.shape[0]  # Total number of trips
if st.sidebar.button("🤔 Do you know the number of bike-sharing trips that took place in London for August 2023?", key="fun_fact_button"):
    st.sidebar.markdown(
        f"""
        <div style='text-align: left; font-size: 18px;'>
            <div style='color: #ADD8E6; font-size: 60px; text-align: center; font-style: italic;'>
                {total_trips:,}
            </div> 
        </div>
        """, 
        unsafe_allow_html=True
        )

    st.sidebar.markdown(
        """
        <div style='text-align: center; font-size: 24px;'>
            <span style="background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet); -webkit-background-clip: text; color: transparent;">
                It's a month packed with adventures!
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Add custom CSS to change the sidebar button format
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] button {
        background-color: #FF69B4;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 10px 0;
        font-family: 'Pacifico', cursive;
        text-align: left;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #FF1493;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Overview ---
if page == 'Overview':
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
             use_column_width=True)

# --- Data Sources ---
elif page == 'Data Sources':
    st.header('Data Sources')

    tab1, tab2 = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset"])

    with tab1:
        st.subheader('🚴 London Bike-Sharing Dataset')
        st.write("""The bike dataset contains detailed records of individual bike trips from the Transport for London (TfL) bike-sharing system. """)
        st.write('**Data Access**: ')
        st.write('[Kaggle Dataset](https://www.kaggle.com/datasets/kalacheva/london-bike-share-usage-dataset)')  
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
            - **`precipitation` (Ratio)**: Total precipitation (rain, showers, snow) sum of the preceding hour in mm.  
            - **`wind_speed_10m` (Ratio)**: Wind speed at 10 meters above ground in m/s.  
            - **`wind_direction_10m` (Circular Numeric)**: Wind direction at 10 meters above ground in degrees.  
            - **`weather_code` (Nominal)**: Weather condition represented by a numeric code.  
             """, unsafe_allow_html=True)
        # Hide the weather code information initially
        if st.button("Show Detailed Weather Code Info", key="weather_code_button"):
            st.image('https://github.com/Leftium/weather-sense/assets/381217/6b346c7d-0f10-4976-bb51-e0a8678990b3', use_column_width=True)
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
        
# --- Data Exploration ---
if page == 'Data Exploration':
    st.header('Data Exploration')
    # Create tabs for datasets
    bike_tab, weather_tab, combined_tab = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "🔍 Combined Insights"])

    with bike_tab:
        try:
            if bike_0 is not None:

                st.subheader('🚴 London Bike-Sharing Dataset')

                with st.expander("🚴 Raw Data", expanded=True):
                    st.subheader('**🚴 Overview**')
                    st.write(bike_0.head(10))  # Display first 10 rows
                    
                    st.divider()

                    col1, col2 = st.columns([1,1.5])
                    
                    with col1:
                        st.write(bike_0.dtypes.apply(lambda x: x.name).to_frame('Type').style.set_table_styles(
                            [{'selector': 'th', 'props': [('text-align', 'left')]},
                            {'selector': 'td', 'props': [('text-align', 'left')]}]
                        ))

                    with col2:
                        if st.checkbox("**Check Missingness**", key="missing_bike"):
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
                                    
                        if st.checkbox("**Check Duplicate**", key="duplicate_bike"):
                                num_duplicates = bike_0.duplicated().sum()
                                if num_duplicates > 0:
                                    st.markdown(f"<div style='color: #5F9EA0;'>Number of duplicate rows: {num_duplicates}</div>", unsafe_allow_html=True)

                                    bike_cleaned = bike_0.drop_duplicates()
                                    st.markdown("<div style='color: #5F9EA0;'>- Duplicates have been removed.</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div style='color: #5F9EA0;'>No duplicates found.</div>", unsafe_allow_html=True)

                        if st.checkbox("**Check Shape**", key="shape_bike"):
                            st.write(f"The original dataset contains:")
                            st.markdown(f"""
                            - **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** columns
                            - Both categorical and numerical features
                            """)

                        if st.checkbox("**Adjustment**", key="data_types_bike"):
                            st.write("Some columns are stored as objects (strings) and need to be adjusted:")
                            st.markdown("""
                            - **`Start date` and `End date`**: Convert to `datetime` format
                            - **`Total duration (ms)`**: Convert from milliseconds to minutes           
                            - **`Total duration`**: Redundant and dropped
                            - **`Bike model`**: Convert to `category` type 
                            """)
                            st.write("Also,")
                            st.markdown("""
                            - Create a **`date`** column in `yyyy-mm-dd HH:MM` based on `Start date` for merging
                            - Check consistency: 
                                - `Start station` vs `Start station number`
                                - `End station` vs `End station number`
                            """)
                                
                with st.expander("🚵 Data Cleaning & Preprocessing", expanded=False):
                    
                    st.subheader('**Data Cleaning & Preprocessing**')
                    bike_1 = bike_0.copy() # Create a copy of the original dataset

                    st.write("**Data Types after Conversion:**")

                    try:
                        # Convert 'Start date' and 'End date' to datetime format
                        bike_1['Start date'] = pd.to_datetime(bike_1['Start date'])
                        bike_1['End date'] = pd.to_datetime(bike_1['End date'])

                        # Extract `Date` from 'Start date'
                        bike_1['Date'] = bike_1['Start date'].dt.floor('T') # Round down to the nearest minute
                        st.write("Date extracted in the consistent `yyyy-mm-dd HH:MM` format for merging.")
                    except Exception as e:
                        st.error(f"Error in converting 'Start date' and 'End date'. Error: {e}")
                   
                    # Convert 'Total duration (ms)' from milliseconds to minutes
                    bike_1['Total duration (m)'] = round(bike_1['Total duration (ms)'] / 60000, 0)

                    # Check the changes
                    st.write(bike_1[['Date', 'Start date', 'End date', 'Total duration (ms)', 'Total duration (m)']].tail())
                    
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.write(bike_1[['Start date', 'End date', 'Date', 'Total duration (m)']].dtypes.apply(lambda x: x.name).to_frame('Type (Corrected)').style.set_table_styles(
                            [{'selector': 'th', 'props': [('text-align', 'left')]},
                            {'selector': 'td', 'props': [('text-align', 'left')]}]
                            ))

                    with col2:
                        st.write(""" 
                        - `Date` was extracted from `Start date` in `yyyy-mm-dd HH:MM` format
                        - `Total duration (m)` was converted to minutes.
                        - Redundant columns were dropped after confirming the changes.
                        """)

                        # Drop redundant/irrelevant columns
                        bike_1.drop(columns=['Number', 'Start date', 'End date', 'Total duration', 'Total duration (ms)', 'Bike number'], inplace=True)
                    
                    st.divider()

                    st.write('**Encoding**')    
                    # Label encoding for bike model
                    if st.checkbox("**Label Encoding for Bike Model**"):
                        st.write("Label encoding is ideal, since `Bike model` contains only two unique values")

                        bike_model_counts = pd.Series({'CLASSIC': 716639, 'PBSC_EBIKE': 59888})

                        # Pie chart for bike model distribution
                        fig, ax = plt.subplots(figsize=(1.5, 1.5))
                        labels = [f'{label} (N={count:,})' for label, count in zip(bike_model_counts.index, bike_model_counts)]
                        
                        wedges, texts, autotext = ax.pie(
                            bike_model_counts, 
                            autopct='%1.1f%%', 
                            startangle=90, 
                            labels=labels, 
                            colors=['skyblue', 'orange'],
                            textprops={'fontsize': 10, 'color': 'black'}
                        )
                        for text in texts + autotext:
                            text.set_fontsize(10)
                        ax.set_title('Bike Model Distribution', fontsize=12)
                        st.pyplot(fig)

                        # Apply label encoding
                        try:
                            le = LabelEncoder()
                            bike_1['Bike model_2'] = le.fit_transform(bike_1['Bike model'])
                            st.write("Label encoding was applied successfully.")
                        except Exception as e:
                            st.error(f"Label encoding failed. Please check the bike model data. Error: {e}")

                        bike_1.drop(columns=['Bike model'], inplace=True)

                    # One-hot encoding for station names
                    if st.checkbox("**One-Hot Encoding for Station Names**"):
                        st.write("One-hot encoding is appropriate, since station names are nominal categorical variables with no inherent order.")

                        col1, col2 = st.columns([1,2])
                        with col1:
                            with st.popover("Data Quality Check"):
                        
                                # Check unique station numbers
                                unique_start_stations = bike_1['Start station number'].nunique()
                                unique_end_stations = bike_1['End station number'].nunique()

                                # Check counts of unique start and end stations
                                counts_start = bike_1['Start station number'].value_counts()
                                counts_end = bike_1['End station number'].value_counts()
                                st.write(f"There are <span style='color: #4682B4;'>{unique_start_stations:,}</span> unique start stations, with a total number of trips per station as <span style='color: #4682B4;'>{counts_start.sum():,}</span>.", unsafe_allow_html=True)
                                st.write(f"There are <span style='color: #4682B4;'>{unique_end_stations:,}</span> unique end stations, with a total number of trips per station as <span style='color: #4682B4;'>{counts_end.sum():,}</span>.", unsafe_allow_html=True)

                                # Check if station name and number is matching
                                matched_start = bike_1.groupby('Start station')['Start station number'].nunique()
                                matched_end = bike_1.groupby('End station')['End station number'].nunique()
                                if (matched_start > 1).any():
                                    st.write("- Mismatched Start Stations:")
                                    st.write(matched_start[matched_start > 1])
                                else:
                                    st.write("- All start station names and numbers match.")
                                if (matched_end > 1).any():
                                    st.write("- Mismatched End Stations:")
                                    st.write(matched_end[matched_end > 1])
                                else:
                                    st.write("- All end station names and numbers match.")
                        
                        with col2:
                            with st.popover("One-hot encoding for station names is deferred at this stage."):
                                st.write(
                                    """While one-hot encoding can be beneficial for machine learning models, the large number of unique stations complicates the dataset and could lead to the curse of dimensionality. Although we could focus on the top 10 or 20 stations, it may be more effective to postpone this step until after EDA and feature selection. Since each station name has a consistent station number, we might find that encoding isn’t necessary."""
                                )

                    st.divider ()

                    st.write('**Transformation & Outlier**')
                    # Check for outliers in 'Total duration (m)'
                    st.write(bike_1['Total duration (m)'].describe().to_frame().T)
                    
                    st.write('**Initial Observations:**')
                    st.write('- Some trips lasting over 124,000 minutes (more than 86 days).')
                    st.write('- std is relatively high, indicating a wide range of trip durations.')

                    # Check for zero values
                    zero_count = (bike_1['Total duration (m)'] == 0).sum()
                    zero_percent = round((zero_count / len(bike_1)) * 100, 2)
                    st.write(f"- There are {zero_count:,} ({zero_percent}%) trips with a duration of less than 1 minute (specifically those under 30 seconds).")
                    st.write("These trips are likely errors or anomalies and should be removed.")
                    bike_1 = bike_1[bike_1['Total duration (m)'] > 0] # Remove trips with duration less than 1 minute
                    
                    # Copy the original dataset for further processing
                    bike_2 = bike_1.copy()

                   # Log Transformation
                    bike_2['Total duration (m)_log'] = np.log1p(bike_2['Total duration (m)'])  
                    # st.write(bike_2['Log Total duration (m)'].describe().to_frame().T)
                    
                    def detect_outliers_iqr(df, column):
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)

                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                        

                    # Outlier Detection using IQR on the log data
                    bike_3 = detect_outliers_iqr(bike_2, 'Total duration (m)_log')
                    st.write(f"Outliers detected using the IQR method: {bike_2.shape[0] - bike_3.shape[0]:,} records removed.")

                    # Min-Max Scaling after outlier removal
                    scaler = MinMaxScaler()
                    bike_3['Total duration (m)_s'] = scaler.fit_transform(bike_3[['Total duration (m)_log']])

                    # Create subplots for 2x2 arrangement
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    plt.subplots_adjust(hspace=0.4, wspace=0.4)

                    # Original Total Duration (m)
                    sns.boxplot(x=bike_2['Total duration (m)'], color='skyblue', ax=axes[0, 0])
                    axes[0, 0].set_title('Original Total Duration (m)')
                    axes[0, 0].set_xlabel('Total Duration (m)')

                    # Log Transformed Total Duration (m)
                    sns.boxplot(x=bike_2['Total duration (m)_log'], color='lightgreen', ax=axes[0, 1])
                    axes[0, 1].set_title('Log Transformed Total Duration (m)')
                    axes[0, 1].set_xlabel('Log Total Duration (m)')

                    # Log Transformed Total Duration (m) with Outlier Removal
                    sns.boxplot(x=bike_3['Total duration (m)_log'], color='lightblue', ax=axes[1, 0])
                    axes[1, 0].set_title('Log Total Duration (m) (Outliers Removed)')
                    axes[1, 0].set_xlabel('Log Total Duration (m)')

                    # Normalized Total Duration (m)
                    sns.boxplot(x=bike_3['Total duration (m)_s'], color='lightcoral', ax=axes[1, 1])
                    axes[1, 1].set_title('Scaled Total Duration (m)')
                    axes[1, 1].set_xlabel('Scaled Total Duration (m)')

                    st.pyplot(fig)

                    bike_3.drop(columns=['Total duration (m)', 'Total duration (m)_log'], inplace=True)

                    # Summary of outliers removed
                    st.write("""
                    - Log transformation was applied to reduce skewness and stabilize variance for a more normal distribution.
                    - MinMax scaling rescaled the log-transformed values to a [0, 1] range for better comparability with other continuous variables.
                    - Outliers were detected using the IQR method and removed.
                    - Yet, depends on the purpose of visualizations/analyses, the log-transformed and scaled values can be used.
                    - {bike_0.shape[0] - bike_3.shape[0]:,} outliers were removed, leaving {bike_3.shape[0]:,} records.
                    """)

                if st.button("Show Cleaned Bike Data Preview"):
                    st.write(bike_3.head())
                    
 
            else:
                st.error("Bike dataset is not loaded properly.")
        except FileNotFoundError:
            st.error("The file '1_LondonBikeJourneyAug2023.csv' was not found.")


    with weather_tab:
        try:
            if weather_0 is not None:
                
                st.subheader('🌤️ London Weather Dataset')

                with st.expander("🌤️ Raw Dataset", expanded=False):
                    st.subheader('**🌤️ Overview**')
                    st.write(weather_0.head(10))  # Display first 10 rows
                                    
                    st.divider()

                    col1, col2 = st.columns([1,1.5])

                    with col1:
                        st.write(weather_0.dtypes.apply(lambda x: x.name).to_frame('Type').style.set_table_styles(
                            [{'selector': 'th', 'props': [('text-align', 'left')]},
                            {'selector': 'td', 'props': [('text-align', 'left')]}]
                        ).set_table_attributes('style="width: auto;"'))

                    with col2:
                        st.write(f"The original dataset contains **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** columns, with both categorical and numerical features.")
                    
                        if st.checkbox("**Check Missingness**", key="missing_weather"):
                            missing_values = weather_0.isnull().sum()
                            if missing_values.sum() == 0:
                                st.markdown("<span style='color: #5F9EA0;'>No missing values found.</span>", unsafe_allow_html=True)
                            else:
                                missing_percent = (missing_values / len(weather_0)) * 100
                                missing_summary = pd.DataFrame({
                                    'Missing Values': missing_values,
                                    '% Missing': missing_percent
                                })
                                st.markdown(missing_summary.style.applymap(lambda x: 'color: red' if x > 0 else 'color: green').render(), unsafe_allow_html=True)
                                
                        if st.checkbox("**Check Shape**", key="shape_weather"):
                            st.write(f"The original dataset contains:")
                            st.markdown(f"""
                            - **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** columns
                            - All numerical features
                            """)

                        if st.checkbox("**Check Duplicate**", key="duplicate_weather"):
                            num_duplicates = weather_0.duplicated().sum()
                            if num_duplicates > 0:
                                st.markdown(f"<div style='color: #5F9EA0;'>Number of duplicate rows: {num_duplicates}</div>", unsafe_allow_html=True)

                                weather_cleaned = weather_0.drop_duplicates()
                                st.markdown("<div style='color: #5F9EA0;'>- Duplicates have been removed.</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='color: #5F9EA0;'>No duplicates found.</div>", unsafe_allow_html=True)

                        if st.checkbox("**Adjust Data Types**", key="data_types_weather"):
                            st.markdown("""
                            - **`date`**: Convert to `datetime` format and remove timezone information
                            - **`weather_code`**: Map to weather descriptions
                            - **`Date`** was extracted from `date` in `yyyy-mm-dd HH:MM` format for merging
                            """)

                with st.expander("🌤️ Data Cleaning & Preprocessing", expanded=False):
                    st.write("**Data Types after Conversion:**")
                    weather_1 = weather_0.copy()

                    # Convert 'date' to datetime format
                    # Convert 'date' to datetime and remove timezone information
                    try:
                        weather_1['date'] = weather_1['date'].dt.tz_localize(None)  # Remove timezone information
                        weather_1['Date'] = weather_1['date'].dt.floor('T') # Round down to the nearest minute

                        # st.write(weather_1['Date'].dtypes)
                        # st.write(weather_1[['date', 'Date']].head())

                        st.write(f"- `Date` extracted in the consistent `yyyy-mm-dd HH:MM` format for merging.")
                    except Exception as e:
                        st.error(f"Error in creating 'Date' column. Error: {e}")

                    # Map weather codes to descriptions

                    # Create a dictionary of weather codes and descriptions
                    weather_code_mapping = {
                        0: "Clear Sky",
                        1: "Partly Cloudy",
                        2: "Cloudy",
                        3: "Overcast",
                        45: "Fog",
                        48: "Rime Fog",
                        51: "Drizzle",
                        53: "Moderate Drizzle",
                        55: "Heavy Drizzle",
                        61: "Slight Rain",
                        63: "Moderate Rain",
                        65: "Heavy Rain",
                        71: "Light Snow",
                        73: "Moderate Snow",
                        75: "Heavy Snow",
                        80: "Rain Showers",
                        95: "Thunderstorm",
                        96: "Thunderstorm with Hail"
                    }

                    weather_1['Weather Description'] = weather_1['weather_code'].map(weather_code_mapping)

                    # Plot the weather code distribution
                    fig, ax = plt.subplots(figsize=(5, 5))
                    sns.countplot(x=weather_1['Weather Description'], palette='viridis', ax=ax)
                    ax.set_title('Weather Description Distribution')
                    ax.set_xlabel('Weather Description')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)

                    # Check the changes
                    # st.write(weather_1[['date', 'Date', 'weather_code', 'Weather Description']].head())

                    # Drop redundant columns
                    weather_1.drop(columns=['date'], inplace=True)

                    st.divider()

                    st.write('**Transformation & Outlier**')
                    
                    weather_2 = weather_1.copy()

                    numeric_columns = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'relative_humidity_2m']
                    st.write(weather_2[numeric_columns].describe().T. round(2))

                    st.write(f"Outliers detected using the IQR method")
                    for variable in numeric_columns:
                        if variable == 'relative_humidity_2m':
                            st.write(f"'{variable}' are already in the reasonable range [0, 100].")

                        else:
                            weather_2 = detect_outliers_iqr(weather_2, variable)
                            st.write(f"- '{variable}': {weather_0.shape[0] - weather_2.shape[0]:,} records removed.")

                    st.write('Min-Max Scaling was applied to normalize the data for better comparability.')

                    variable = st.selectbox("Select a variable to visualize", numeric_columns)
                    
                    for variable in numeric_columns:
                        # Min-Max Scaling
                        scaler = MinMaxScaler()
                        weather_2[f'{variable}_n'] = scaler.fit_transform(weather_2[[variable]])
                    
                    if variable:    
                        # Before Normalization
                        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                        sns.boxplot(x=weather_1[variable], color='lightblue', ax=axes[0])
                        axes[0].set_title(f'{variable} Before Normalization')
                        axes[0].set_xlabel(variable)

                        # After Normalization
                        sns.boxplot(x=weather_2[f'{variable}_n'], color='lightgreen', ax=axes[1])
                        axes[1].set_title(f'{variable} After Normalization')
                        axes[1].set_xlabel(f'{variable}_n')

                        st.pyplot(fig)

                    st.write('**Check Changes**')
                    st.write(weather_2[[variable, f'{variable}_n']].describe().round(2))
                    
                    weather_2.drop(columns=numeric_columns, inplace=True)
               
                with st.expander("🌤️ Data Summary aftr Cleaning", expanded=True):
                    st.write(weather_2.drop(columns=['weather_code', 'Date']).describe().T.round(2))

            else:
                st.error("Weather dataset is not loaded properly.")
        except FileNotFoundError:
            st.error("The file '2_london_weather_2023.csv' was not found.")


        with combined_tab:

            if bike_3 is not None and weather_2 is not None:

                # Merge the bike and weather datasets
                combined = pd.merge(bike_3, weather_2, on='Date', how='inner')
                
                st.subheader('🔍 Combined Insights')
                #st.write('**The first few rows of the combined dataset:**')
                #st.write(combined.head())

                # Extract time variants from 'Date'
                st.write("Generate Time-Related Features")
                # Extract time variants from 'Date'
                combined['Day_of_Month'] = combined['Date'].dt.day  # 1-31
                combined['Day_of_Week'] = combined['Date'].dt.dayofweek + 1  # Monday=1, Sunday=7
                combined['Hour_of_Day'] = combined['Date'].dt.hour  # 0-23
                combined['is_Weekend'] = combined['Day_of_Week'].apply(lambda x: 1 if x > 5 else 0)  # 1=Weekend, 0=Weekday
                # combined['Time_of_Day'] = combined['Hour_of_Day'].apply(
                #    lambda x: 'Morning' if 5 <= x < 12 else 'Afternoon' if 12 <= x < 17 else 'Evening' if 17 <= x < 21 else 'Night'
                #)

                # Check the changes
                st.write(combined[['Date', 'Day_of_Month', 'Day_of_Week', 'Hour_of_Day', 'is_Weekend']].sample(5))
               
                # Convert 'Total duration (m)' to categorical (maybe later)

                # Renaming the specified columns
                combined.rename(columns={
                    "Total duration (m)_s": "Total_duration_min",
                    "temperature_2m_n": "real_temperature_C",
                    "apparent_temperature_n": "feels_like_temperature_C",
                    "wind_speed_10m_n": "wind_speed_10m",
                    "relative_humidity_2m_n": "humidity_percentage"
                }, inplace=True)

                if st.button ("Final Data View"):
                    st.dataframe(combined)
                
                st.write("Summary Statistics")
                numeric_columns = ['Total_duration_min', 'real_temperature_C', 'feels_like_temperature_C', 'wind_speed_10m', 'humidity_percentage']
                st.write(combined[numeric_columns].describe().T.round(2))
                st.write(combined.dtypes)

                # Save the combined dataset
                combined.to_csv('datasets/3_london_bike_weather_2023.csv', index=False)

# --- Data Visualization ---

elif page == 'Data Visualization':

    def load_combined_data():
        return pd.read_csv('3_london_bike_weather_2023.csv')
    
    combined = load_combined_data()

    # Define groups for visualization
    weather_cols = ['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']
    time_cols = ['Day_of_Month', 'Day_of_Week', 'Hour_of_Day']
    numeric_cols = weather_cols + time_cols + ['Total_duration_min']

    st.header('Data Visualization')

    eda_option = st.selectbox(
        "Pick one to explore:",
        [
            "What correlations exist between variables?",
            "How do bike-sharing trends change over time?",
            "What impact does weather have on bike-sharing?",
            "Which stations are more popular?"
        ]
    )

    # Add custom CSS to change the selectbox format
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] {
            background-color: #f0f8ff;
            color: #00008b;
            font-size: 18px;
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Arial', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    if eda_option == "What correlations exist between variables?":

        plt.figure(figsize=(12, 8))
        color_scales = 'Viridis'

        fig_corr = px.imshow(
            combined[numeric_cols].corr(),
            text_auto='.2f',  # 2 decimal places
            aspect="auto",
            color_continuous_scale=color_scales,
            labels=dict(color="Correlation"),
            title="Correlation Heatmap: Overall"
        )
        fig_corr.update_traces(hovertemplate='Corr(%{x}, %{y}) = %{z:.6f}<extra></extra>')  # Show 6 decimal places on hover
        st.plotly_chart(fig_corr)
        plt.title("Correlation Heatmap: Overall", fontsize=16)

        st.markdown("""
        **Observation:**
        - **Temperature** has the strongest correlation with bike-sharing.
        - **Temperature** and **humidity** exhibit a negative correlation.
        - **Time variables** show weak correlations.

        Understanding the relationship between weather, especially temperature, and bike-sharing is crucial for predicting bike-sharing trends.
        """)

    if eda_option == "How do bike-sharing trends change over time?":

        time_variable = st.selectbox(
            "Select a type to visualize:",
            ["Bike-Sharing Trends", "Weekend Effect", "Average by Hour and Day of Week"]
        )

        if time_variable == "Bike-Sharing Trends":
            selected_time = st.radio(
                "**Select a time variable for trends:**",
                time_cols,
                horizontal=True
            )
            if selected_time:
               
                # Count the number of bike-sharing by the selected time variable
                counts = combined.groupby(selected_time).size().reset_index(name='Number of Bike-Sharing')
                
                # Create a line plot to show bike-sharing trends
                fig_time = px.line(
                    counts,
                    x=selected_time,
                    y='Number of Bike-Sharing',
                    title=f"Bike-Sharing Trends by {selected_time}",
                    labels={selected_time: selected_time, 'Number of Bike-Sharing': 'Count'},
                    markers=True,
                    line_shape='linear'
                )

                fig_time.update_traces(mode='lines+markers', line=dict(color='green'))
                fig_time.add_scatter(
                    x=counts[selected_time],
                    y=counts['Number of Bike-Sharing'],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.2)',
                    line=dict(width=0)
                )
                fig_time.update_layout(hovermode='x unified')

                st.plotly_chart(fig_time)
            
        elif time_variable == "Weekend Effect":
            # Hourly changes in bike-sharing trends (Weekday vs Weekend)
            # Count rentals for weekends
            weekend_counts = combined[combined['is_Weekend'] == 1].groupby('Hour_of_Day').size().reset_index(name='Number of Bike-Sharing')
            weekend_counts['Type'] = 'Weekend'  # Add a column to indicate Weekend

            # Count bike-sharing for weekdays
            weekday_counts = combined[combined['is_Weekend'] == 0].groupby('Hour_of_Day').size().reset_index(name='Number of Bike-Sharing')
            weekday_counts['Type'] = 'Weekday'  # Add a column to indicate Weekday

            # Combine both weekend and weekday counts
            combined_counts = pd.concat([weekend_counts, weekday_counts])

            # Create line plot to show hourly changes for weekends and weekdays
            fig_hourly = px.line(
                combined_counts,
                x='Hour_of_Day',
                y='Number of Bike-Sharing',
                color='Type', 
                title="Hourly Bike-Sharing: Weekday vs Weekend",
                labels={"Hour_of_Day": "Hour of Day", "Number of Bike-Sharing": "Count"},
                markers=True,
                color_discrete_map={'Weekday': 'green', 'Weekend': 'lightgreen'}  # Use green shades for the lines
            )

            fig_hourly.update_traces(mode='lines+markers', hovertemplate='N = %{y}')
            fig_hourly.update_layout(hovermode='x unified')
            st.plotly_chart(fig_hourly)

            st.write("""
            **Observation:**
            - More bike-sharing occurs on **weekdays** than on **weekends**.
            - **Weekdays** show a clear peak in bike-sharing during the morning and evening rush hours.
            - **Weekends** have a more even distribution throughout the day, with a slight peak in the afternoon.
            """)

        elif time_variable == "Average by Hour and Day of Week":

            # Create a mapping dictionary
            day_mapping = {
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday"
            }

            combined['Day_Name'] = combined['Day_of_Week'].map(day_mapping)
            
            # Group and unstack to create a 2D matrix
            avg_hour_day = combined.groupby(['Hour_of_Day', 'Day_Name']).size().unstack(fill_value=0)

            # Reorder the columns to ensure the days of the week are in the correct order
            avg_hour_day = avg_hour_day[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

            # Create a heatmap
            fig_avg_hour_day = px.imshow(
                avg_hour_day,
                aspect="auto",
                labels=dict(x="Day of Week", y="Hour of Day", color="Average Bike-Sharing"),
                title="Average Bike-Sharing by Hour and Day of Week",
                color_continuous_scale='YlGnBu',
                text_auto='.0f'  # Display annotations with 0 decimal places
            )
            fig_avg_hour_day.update_layout(
                autosize=True
            )

            # Update layout for axis titles
            fig_avg_hour_day.update_layout(xaxis_title='Day of Week', yaxis_title='Hour of Day')
            st.plotly_chart(fig_avg_hour_day)

            st.markdown("""
            **Observation:**
            - The heatmap shows the average number of bike-sharing by hour and day of the week.
            - **Peak bike-sharing** occurs on **weekdays**, especially in the **late afternoons (4 PM - 7 PM)**. This suggests that commuters likely use bikes after work.
            - **Wednesdays and Thursdays** have the highest activity, while **Sundays** have the lowest.
            """)

    if eda_option == "What impact does weather have on bike-sharing?":
        

        if st.button("Number of Bike-Sharing Trips by Weather Condition"):

            # Group by the chosen weather condition (e.g., weather_code)
            weather_counts = combined.groupby('Weather Description').size().reset_index(name='Number of Bike-Sharing')

            # Create a bar plot for the number of bike-sharing trips by weather condition
            fig_bar = px.bar(
                weather_counts,
                x='Weather Description',  
                y='Number of Bike-Sharing',
                labels={'weather_code': 'Weather Condition', 'Number of Bike-Sharing': 'Count'},
                color='Number of Bike-Sharing',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig_bar)

            st.markdown("""
            **Observation:**
            - The bar plot shows the number of bike-sharing trips by weather condition.
            - **Clear Sky** and **Partly Cloudy** weather conditions have the highest number of bike-sharing trips.
            - **Thunderstorms** and **Heavy Snow** have the lowest number of bike-sharing trips.
            """)

        st.divider()
        # Group and unstack to create a 2D matrix
        avg_hour_weather = combined.groupby(['Hour_of_Day', 'Weather Description']).size().unstack(fill_value=0)

        # Create a heatmap
        fig_avg_hour_weather = px.imshow(
            avg_hour_weather,
            aspect="auto",
            labels=dict(y="Weather Condition", x="Hour of Day", color="Average Bike-Sharing"),
            title="Average Bike-Sharing by Hour and Weather Condition",
            color_continuous_scale='Blues',
            text_auto='.0f'  # Display annotations with 0 decimal places
        )
        fig_avg_hour_weather.update_layout(
            autosize=True
        )

        # Update layout for axis titles
        fig_avg_hour_weather.update_layout(xaxis_title='Weather Condition', yaxis_title='Hour of Day')
        st.plotly_chart(fig_avg_hour_weather)

        # Group and unstack to create a 2D matrix for average bike-sharing by day and weather condition
        avg_day_weather = combined.groupby(['Weather Description', 'Day_of_Week']).size().unstack(fill_value=0)

        # Create a heatmap
        fig_avg_day_weather = px.imshow(
            avg_day_weather,
            aspect="auto",
            labels=dict(y="Weather Condition", x="Day of Week", color="Average Bike-Sharing"),
            title="Average Bike-Sharing by Day and Weather Condition",
            color_continuous_scale='Blues',
            text_auto='.0f'  # Display annotations with 0 decimal places
        )
        fig_avg_day_weather.update_layout(
            autosize=True
        )

        # Update layout for axis titles
        fig_avg_day_weather.update_layout(xaxis_title='Day of Week', yaxis_title='Weather Condition')
        st.plotly_chart(fig_avg_day_weather)

        st.write("""
        **Observation:**
        - The heatmap shows the average number of bike-sharing by day and weather condition.
        - **Clear Sky** and **Partly Cloudy** weather conditions have the highest average bike-sharing.""")


    if eda_option == "Which stations are more popular?":
       
        # Station selection
        station_option = st.radio("Select a station type:", ["Start station", "End station"])

        # Determine unique stations
        unique_stations = combined[station_option].nunique()

        # Slider for selecting the number of top stations to display
        top_n = st.slider("Select the number of top stations to display:", min_value=1, max_value=unique_stations, value=10, step=1)

        # Identify the top stations based on the selected station type
        if station_option == "Start Station":
            top_stations = combined['Start station'].value_counts().head(top_n).index
            top_station_data = combined[combined['Start station'].isin(top_stations)]
        else:
            top_stations = combined['End station'].value_counts().head(top_n).index
            top_station_data = combined[combined['End station'].isin(top_stations)]

        # Create a bar plot for the top N stations
        fig_station = px.bar(
            top_station_data.groupby(station_option).size().reset_index(name='Number of Bike-Sharing').sort_values(by='Number of Bike-Sharing', ascending=False),
            x='Number of Bike-Sharing',  # Count on the x-axis
            y=station_option,             # Station names on the y-axis
            title=f"Top {top_n} {station_option}s",
            labels={station_option: f"{station_option} Name", "Number of Bike-Sharing": "Count"},
            color=station_option,          # Color by station name
        )

        # Hide the legend
        fig_station.update_layout(showlegend=False)

        # Update layout for the color bar
        fig_station.update_layout(coloraxis_colorbar=dict(title='Count'))

        # Display the plot in Streamlit
        st.plotly_chart(fig_station)

        # Calculate the percentage of trips for the top stations
        total_trips = combined.shape[0]
        top_station_trips = top_station_data.shape[0]
        percentage = (top_station_trips / total_trips) * 100

        st.write(f"These top {top_n} {station_option.lower()}s represent {percentage:.2f}% of the total bike-sharing trips.")
        
        if st.button("Show Top Stations"):
            st.write(top_station_data.head())
        
        # Function to plot time distribution based on 'Day_of_Month'
        def plot_time_distribution(data, title, color):
            plt.figure(figsize=(10, 5))
            sns.histplot(data['Day_of_Month'], kde=True, color=color, bins=31)
            plt.title(title)
            plt.xlabel('Day of Month')
            plt.ylabel('Frequency')
            st.pyplot(plt.gcf())  # Display the plot in Streamlit

        # Create figure for subplots
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))

        # Function to plot time distribution based on 'Day_of_Month'
        def plot_time_distribution(data, title, color):
            plt.figure(figsize=(10, 5))
            sns.histplot(data['Day_of_Month'], kde=True, color=color, bins=31)
            plt.title(title)
            plt.xlabel('Day of Month')
            plt.ylabel('Frequency')
            st.pyplot(plt.gcf())  # Display the plot in Streamlit

        # Filter data based on selected station type and top stations
        filtered_data = top_station_data[top_station_data[station_option].isin(top_stations)]

        # Plot the distribution for the selected top stations
        plot_time_distribution(filtered_data, f'Distribution of {station_option} Times for Top Stations', 'green' if station_option == "Start Station" else 'red')

        st.markdown("""
        **Observation:**
        - Identifying the top stations reaveals popular bike-sharing locations.
        - The time distribution of bike-sharing provides insights into peak hours and usage patterns.
        - This information can help optimize bike availability and station capacity.
        - Given the large number of unique stations, focusing on regional or cross-regional patterns may be more beneficial.
        """)


