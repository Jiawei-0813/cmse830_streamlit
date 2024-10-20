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

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


st.title('London Bike-Sharing Trends')
# st.subheader('CMSE 830 Midterm Project')

# --- Load Data ---
# Load bike-sharing data
@st.cache_data
def load_data():
    bike_0 = pd.read_csv('1_Midterm/datasets/1_LondonBikeJourneyAug2023.csv')
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
    weather_0.to_csv('1_Midterm/datasets/2_london_weather_2023.csv', index=False)
    return weather_0

weather_0 = fetch_and_save_weather_data()
    
# --- Sidebar ---
# Display menu
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go To', 
                        ['Overview',
                         'Data Sources', 
                         'Data Exploration', 
                         'Data Visualization'
                         ])

# Fun Fact
total_trips = bike_0.shape[0]  # Total number of trips
if st.sidebar.button("🤔 Guess how many bike-sharing trips happened in London in August 2023?", key="fun_fact_button"):
    st.sidebar.markdown(
        f"""
        <div style='text-align: center; font-size: 18px;'>
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
            <span style="background: linear-gradient(to right, #FF69B4, #FF1493); -webkit-background-clip: text; color: transparent;">
                Wow! That's a lot of adventures on two wheels!
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
        text-align: center;
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
        
# --- Data Inspection ---
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
                    st.dataframe(bike_0.head())  
                    
                    st.divider()

                    col1, col2, col3 = st.columns([1,1.5, 1.5])
                    
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

                        if st.checkbox("**Check Shape**", key="shape_bike"):
                            st.write(f"The original dataset contains:")
                            st.markdown(f"""
                            - **{bike_0.shape[0]:,}** rows and **{bike_0.shape[1]}** columns
                            - Both categorical and numerical features
                            """)

                    with col3:                
                        if st.checkbox("**Check Duplicate**", key="duplicate_bike"):
                                num_duplicates = bike_0.duplicated().sum()
                                if num_duplicates > 0:
                                    st.markdown(f"<div style='color: #5F9EA0;'>Number of duplicate rows: {num_duplicates}</div>", unsafe_allow_html=True)

                                    bike_cleaned = bike_0.drop_duplicates()
                                    st.markdown("<div style='color: #5F9EA0;'>- Duplicates have been removed.</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div style='color: #5F9EA0;'>No duplicates found.</div>", unsafe_allow_html=True)

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
                        st.write("Since `Bike model` contains only two unique values, label encoding is appropriate.")

                        bike_model_counts = pd.Series({'CLASSIC': 716639, 'PBSC_EBIKE': 59888})
                        bike_model_df = bike_model_counts.reset_index()
                        bike_model_df.columns = ['Bike Model', 'Count']


                        # Pie chart for bike model distribution
                        fig_pie = px.pie(
                            bike_model_df,
                            names='Bike Model',
                            values='Count',
                            title='Bike Model Distribution',
                            hole=0.5,
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hover_data=['Count'],  # Show count on hover
                            color_discrete_map={'CLASSIC': '#1f77b4', 'PBSC_EBIKE': '#ff7f0e'}
                        )
                        fig_pie.update_layout(title={'text': 'Bike Model Distribution', 'x': 0.5, 'xanchor': 'center'})
                        fig_pie.update_traces(textinfo='percent+label', textfont_size=12)
                        fig_pie.update_layout(showlegend=False)
                        st.plotly_chart(fig_pie)

                        # Apply label encoding
                        try:
                            le = LabelEncoder()
                            bike_1['Bike model_2'] = le.fit_transform(bike_1['Bike model'])
                            st.write("Label encoding was applied successfully [Classic = 0].")

                            # st.write(bike_1[['Bike model', 'Bike model_2']].head().T)
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

                    st.write('**Transformation & Outlier Detection**')
                    # Check for outliers in 'Total duration (m)'
                    st.write(bike_1['Total duration (m)'].describe().to_frame().T)
                    
                    st.write('**Initial Observations:**')
                    st.write('- Some trips lasting over 124,000 minutes (more than 86 days).')
                    st.write('- std is relatively high and much higher than the mean, indicating the data is widely spread out, with some trips having durations significantly longer than the average')

                    # Check for zero values
                    zero_count = (bike_1['Total duration (m)'] == 0).sum()
                    zero_percent = round((zero_count / len(bike_1)) * 100, 2)
                    st.write(f"- There are {zero_count:,} ({zero_percent}%) trips with a duration of less than 1 minute (specifically those under 30 seconds). These trips are likely errors or anomalies and should be removed.")
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

                    # Create subplots for 2x2 arrangement using Plotly
                    fig = make_subplots(rows=2, cols=2, subplot_titles=(
                        'Original Total Duration (m)', 
                        'Log Transformed Total Duration (m)', 
                        'Log Total Duration (m) (Outliers Removed)', 
                        'Scaled Total Duration (m)'
                    ))

                    # Original Total Duration (m)
                    fig.add_trace(
                        px.box(bike_2, x='Total duration (m)', color_discrete_sequence=['skyblue']).data[0],
                        row=1, col=1
                    )

                    # Log Transformed Total Duration (m)
                    fig.add_trace(
                        px.box(bike_2, x='Total duration (m)_log', color_discrete_sequence=['lightgreen']).data[0],
                        row=1, col=2
                    )

                    # Log Transformed Total Duration (m) with Outlier Removal
                    fig.add_trace(
                        px.box(bike_3, x='Total duration (m)_log', color_discrete_sequence=['lightblue']).data[0],
                        row=2, col=1
                    )

                    # Normalized Total Duration (m)
                    fig.add_trace(
                        px.box(bike_3, x='Total duration (m)_s', color_discrete_sequence=['lightcoral']).data[0],
                        row=2, col=2
                    )

                    # Update layout
                    fig.update_layout(height=800, width=1000, title_text="Total Duration Analysis")
                    st.plotly_chart(fig)

                    bike_3.drop(columns=['Total duration (m)', 'Total duration (m)_s'], inplace=True)

                    # Summary of outliers removed
                    st.write(f"""
                    - Log transformation was applied to reduce skewness and stabilize variance for a more normal distribution, though some extreme values remained.
                    - Instead of purely relying on mathematical outlier removal, further exploration of these extreme values might offer valuable insights (e.g., unusually long rides, special events).
                    - MinMax scaling was considered to rescale the log-transformed values to a [0, 1] range for better comparability with other continuous variables, but was not used in this case, as scaling is not essential for most EDA visualizations.
                    - Depending on the purpose of visualizations/analyses, the log-transformed values (with or without outliers) will be used for further analysis.
                    - {bike_2.shape[0] - bike_3.shape[0]:,} outliers were removed, leaving {bike_3.shape[0]:,} records.
                    """)

                if st.button("Show Cleaned Bike Data Preview"):
                    st.write(bike_3.head())
                    
            else:
                st.error("Bike dataset is not loaded properly.")
        except FileNotFoundError:
            st.error("The file '1_Midterm//1_LondonBikeJourneyAug2023.csv' was not found.")


    with weather_tab:
        try:
            if weather_0 is not None:
                
                st.subheader('🌤️ London Weather Dataset')

                with st.expander("🌤️ Raw Dataset", expanded=True):
                    st.subheader('**🌤️ Overview**')
                    st.dataframe(weather_0.head())  
                                    
                    st.divider()

                    col1, col2, col3 = st.columns([1,1.5, 1.5])

                    with col1:
                        st.write(weather_0.dtypes.apply(lambda x: x.name).to_frame('Type').style.set_table_styles(
                            [{'selector': 'th', 'props': [('text-align', 'left')]},
                            {'selector': 'td', 'props': [('text-align', 'left')]}]
                        ).set_table_attributes('style="width: auto;"'))

                    with col2:
                       
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

                        if st.checkbox("**Check Duplicate**", key="duplicate_weather"):
                            num_duplicates = weather_0.duplicated().sum()
                            if num_duplicates > 0:
                                st.markdown(f"<div style='color: #5F9EA0;'>Number of duplicate rows: {num_duplicates}</div>", unsafe_allow_html=True)

                                weather_cleaned = weather_0.drop_duplicates()
                                st.markdown("<div style='color: #5F9EA0;'>- Duplicates have been removed.</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='color: #5F9EA0;'>No duplicates found.</div>", unsafe_allow_html=True)

                    with col3:            
                        if st.checkbox("**Check Shape**", key="shape_weather"):
                            st.write(f"The original dataset contains:")
                            st.markdown(f"""
                            - **{weather_0.shape[0]:,}** rows and **{weather_0.shape[1]}** columns
                            - All numerical features
                            """)

                        if st.checkbox("**Adjust Data Types**", key="data_types_weather"):
                            st.markdown("""
                            - **`date`**: Remove timezone information
                            - **`weather_code`**: Map to weather descriptions
                            - **`Date`** was extracted from `date` in `yyyy-mm-dd HH:MM` format for merging
                            """)

                with st.expander("🌤️ Data Cleaning & Preprocessing", expanded=False):
                    st.subheader('**Data Cleaning & Preprocessing**')
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

                    # Count the occurrences of each weather description
                    weather_counts = weather_1['Weather Description'].value_counts().reset_index() # Count the occurrences of each weather description
                    weather_counts.columns = ['Weather Description', 'Count']

                    # Create an interactive bar plot using Plotly
                    fig = px.bar(
                        weather_counts,
                        x='Weather Description',
                        y='Count',
                        title='Weather Description Distribution',
                        labels={'Weather Description': 'Weather Condition', 'Count': 'Number of Occurrences'},
                        color='Count',
                        color_continuous_scale='Viridis'
                    )

                    # Check the changes
                    # st.write(weather_1[['date', 'Date', 'weather_code', 'Weather Description']].head())

                    # Drop redundant columns
                    weather_1.drop(columns=['date'], inplace=True)

                    st.divider()

                    st.write('**Transformation & Outlier Detection**')
                    
                    weather_2 = weather_1.copy()

                    numeric_columns = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'relative_humidity_2m']
                    st.write(weather_2[numeric_columns].describe().T. round(2))

                    st.write(f"Outliers detected using the IQR method: ")
                    for variable in numeric_columns:
                        if variable == 'relative_humidity_2m':
                            st.write(f"`{variable}` are already in the reasonable range [0, 100].")

                        else:
                            weather_2 = detect_outliers_iqr(weather_2, variable)
                            st.write(f"- `{variable}`: {weather_0.shape[0] - weather_2.shape[0]:,} records removed.")

                    st.write('Min-Max Scaling was applied to normalize the data for better comparability.')

                    variable = st.selectbox("Select a variable to visualize", numeric_columns)
                    
                    for variable in numeric_columns:
                        # Min-Max Scaling
                        scaler = MinMaxScaler()
                        weather_2[f'{variable}_n'] = scaler.fit_transform(weather_2[[variable]])
                    
                    if variable:    
                        # Before Normalization
                        fig_before = px.box(weather_1, x=variable, title=f'{variable} Before Normalization', color_discrete_sequence=['lightblue'])
                        fig_before.update_layout(xaxis_title=variable, yaxis_title='Value')

                        # After Normalization
                        fig_after = px.box(weather_2, x=f'{variable}_n', title=f'{variable} After Normalization', color_discrete_sequence=['lightgreen'])
                        fig_after.update_layout(xaxis_title=f'{variable}_n', yaxis_title='Value')

                        # Combine the plots side-by-side
                        fig_combined = make_subplots(rows=1, cols=2, subplot_titles=(f'{variable} Before Normalization', f'{variable} After Normalization'))

                        fig_combined.add_trace(fig_before['data'][0], row=1, col=1)
                        fig_combined.add_trace(fig_after['data'][0], row=1, col=2)

                        fig_combined.update_layout(height=600, width=1200, showlegend=False)

                        # Display the combined plot
                        st.plotly_chart(fig_combined)


                    st.write('**Check Changes**')
                    st.write(weather_2[[variable, f'{variable}_n']].describe().T. round(2))
                    
                    weather_2.drop(columns=numeric_columns, inplace=True)
                    
            if st.button("Show Cleaned Weather Data Summary"):
                st.write(weather_2.drop(columns=['weather_code', 'Date']).describe().T.round(2))
                         
            else:
                st.error("Weather dataset is not loaded properly.")
        except FileNotFoundError:
            st.error("The file '1_Midterm/2_london_weather_2023.csv' was not found.")


        with combined_tab:

            if bike_3 is not None and weather_2 is not None:

                # Merge the bike and weather datasets
                combined = pd.merge(bike_3, weather_2, on='Date', how='inner')
                
                st.subheader('🔍 Combined')
                #st.write('**The first few rows of the combined dataset:**')
                #st.write(combined.head())

                # Extract time variants from 'Date'
                st.write("**Generate Time-Related Features**")
                # Extract time variants from 'Date'
                combined['Day_of_Month'] = combined['Date'].dt.day  # 1-31
                combined['Day_of_Week'] = combined['Date'].dt.dayofweek + 1  # Monday=1, Sunday=7
                combined['Hour_of_Day'] = combined['Date'].dt.hour  # 0-23
                def is_weekend(day):
                    return 1 if day > 5 else 0

                combined['is_Weekend'] = combined['Day_of_Week'].apply(is_weekend)  # 1=Weekend, 0=Weekday

                def categorize_time_of_day(hour):
                    if 5 <= hour < 12:
                        return 'Morning'
                    elif 12 <= hour < 17:
                        return 'Afternoon'
                    elif 17 <= hour < 21:
                        return 'Evening'
                    else:
                        return 'Night'

                combined['Time_of_Day'] = combined['Hour_of_Day'].apply(categorize_time_of_day)

                # Mapping for Day_of_Week to show day names
                day_of_week_mapping = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
                combined['Day_of_Week_Name'] = combined['Day_of_Week'].map(day_of_week_mapping)

                # Mapping for is_Weekend to show name
                weekend_mapping = {0: "Weekday", 1: "Weekend"}
                combined['is_Weekend_Name'] = combined['is_Weekend'].map(weekend_mapping)

                # Check the changes
                st.write(combined[['Date', 'Day_of_Month', 'Day_of_Week', 'Hour_of_Day', 'Time_of_Day', 'is_Weekend']].sample(5))
               
                # Convert 'Total duration (m)' to categorical (maybe later)

                # Renaming the specified columns
                combined.rename(columns={
                    "Total duration (m)_log": "Total_duration_min",
                    "temperature_2m_n": "real_temperature_C",
                    "apparent_temperature_n": "feels_like_temperature_C",
                    "wind_speed_10m_n": "wind_speed_10m",
                    "relative_humidity_2m_n": "humidity_percentage"
                }, inplace=True)

                st.subheader('**Final Data View**')
                st.dataframe(combined)

                cols1, cols2 = st.columns([1, 1])
                with cols1:
                    if st.checkbox ("Summary Statistics"):
                        numeric_columns = ['Total_duration_min', 'real_temperature_C', 'feels_like_temperature_C', 'wind_speed_10m', 'humidity_percentage']
                        st.write(combined[numeric_columns].describe().T.round(2))
                with cols2:
                    if st.checkbox ("Data Types"):
                        st.write (combined.dtypes.apply(lambda x: x.name).to_frame('Type').style.set_table_styles(
                                [{'selector': 'th', 'props': [('text-align', 'left')]},
                                {'selector': 'td', 'props': [('text-align', 'left')]}]
                            ).set_table_attributes('style="width: auto;"'))

                # Save the combined dataset
                combined.to_csv('1_Midterm/datasets/3_london_bike_weather_2023.csv', index=False)

# --- Data Visualization ---

elif page == 'Data Visualization':

    def load_combined_data():
        return pd.read_csv('1_Midterm/datasets/3_london_bike_weather_2023.csv')
    
    combined = load_combined_data()

    # Define groups for visualization
    weather_cols = ['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']
    time_cols = ['Day_of_Month', 'Day_of_Week', 'Hour_of_Day'] # numeric
    numeric_cols = ['Total_duration_min', 'weather_code', 'real_temperature_C', 
                'feels_like_temperature_C', 'wind_speed_10m', 'humidity_percentage', 
                'Day_of_Month', 'Hour_of_Day']

    st.header('Data Visualization')

    eda_option = st.selectbox(
        "Pick one to explore:",
        [
            "What correlations exist between variables?",
            "How do bike-sharing trends change over time?",
            "What impact does weather have on bike-sharing?",
            "Which stations are more popular?",
            "What do we know by now?"
            ""
        ]
    )

    # Add custom CSS to change the selectbox format
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] {
            background-color: #f0f8ff;
            color: #00008b;
            font-size: 28px; 
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Arial', sans-serif;
        }
        div[data-testid="stSelectbox"] select {
            background-color: #e6f2ff; 
            color: #00008b;
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
        plt.title("Correlation Heatmap: Overall", fontsize=26)

        st.markdown("""
        **Observation:**
        - **Temperature** shows a strong positive correlation with bike-sharing patterns.
        - **Real Temperature** and **Humidity** exhibit a noticeable negative correlation, meaning that as humidity increases, temperature tends to decrease.
        - **Time variables** like **Hour of Day** and **Day of Month** show weaker correlations with weather variables and bike-sharing.
        
        Understanding the relationship between weather, especially temperature and humidity, is key to predicting bike-sharing trends.
        """)

    if eda_option == "How do bike-sharing trends change over time?":

        time_variable = st.selectbox(
            "Select one to visualize:",
            ["General Trends", "Time of Day Effect", 
             "Weekday vs Weekend Patterns", "Hourly Patterns by Day"]
        )

        if time_variable == "General Trends":
            selected_time = st.radio(
                "**Select a time variable for trends:**",
                time_cols[::-1],  # Reverse the order of time_cols
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
                    title=f"Bike-Sharing by {selected_time}",
                    labels={selected_time: selected_time, 'Number of Bike-Sharing': 'Count'},
                    markers=True,
                    line_shape='linear'
                )
                fig_time.update_traces(mode='lines+markers', line=dict(color='green'), name='Bike-Sharing Count')
                fig_time.add_scatter(
                    x=counts[selected_time],
                    y=counts['Number of Bike-Sharing'],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(144, 238, 144, 0.2)',  # Light green fill color
                    line=dict(width=0),
                    name='Filled Area'
                )
                fig_time.update_layout(hovermode='x unified')

                # Calculate the total count of bike-sharing
                total_count = counts['Number of Bike-Sharing'].sum()
                # Calculate the percentage of the total for each count
                counts['Percentage'] = counts['Number of Bike-Sharing'] / total_count * 100

                # Mark the maximum y value
                max_y = counts['Number of Bike-Sharing'].max()
                max_x = counts.loc[counts['Number of Bike-Sharing'].idxmax(), selected_time] # Get the corresponding x value
                max_percentage = counts.loc[counts['Number of Bike-Sharing'].idxmax(), 'Percentage'] # Get the corresponding percentage

                if selected_time == 'Day_of_Week':
                    day_of_week_mapping = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
                    max_x_label = day_of_week_mapping[max_x]
                elif selected_time == 'Day_of_Month':
                    max_x_label = f'Aug {max_x}'
                else:
                    max_x_label = max_x

                # Add a marker for the maximum value
                fig_time.add_scatter(
                    x=[max_x],
                    y=[max_y],
                    mode='markers+text',
                    marker=dict(color='darkgreen', size=10),  # Dark green marker color
                    text=[f'{max_x_label}<br>{max_y:,} ({max_percentage:.2f}%)'],
                    textposition='top center',
                    name='Max Value'
                )

                # Update x-axis labels for Day_of_Week
                if selected_time == 'Day_of_Week':
                    fig_time.update_xaxes(
                        tickmode='array',
                        tickvals=[1, 2, 3, 4, 5, 6, 7],
                        ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    )
                # Format y-axis numbers as x,xxx
                fig_time.update_yaxes(tickformat=',')

                # Update the hover template to display the count and percentage
                fig_time.update_traces(hovertemplate='%{y:,} (%{customdata[0]:.2f}%)<extra></extra>', customdata=counts[['Percentage']].values)
                
                st.plotly_chart(fig_time)

        elif time_variable == "Time of Day Effect":
            # Group by day of month and time of day
            time_of_day_counts_by_day = combined.groupby(['Day_of_Month', 'Time_of_Day']).size().reset_index(name='Number of Bike-Sharing')

            # Create a line plot with time of day as hue and day of month as x-axis
            fig_time_of_day = px.line(
                time_of_day_counts_by_day,
                x='Day_of_Month',
                y='Number of Bike-Sharing',
                color='Time_of_Day',  # This will act as the hue
                title="Bike-Sharing by Day of Month and Time of Day",
                labels={"Day_of_Month": "Day of Month", "Number of Bike-Sharing": "Count"},
                markers=True,
                category_orders={"Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"]}  # Set hue order
            )

            fig_time_of_day.update_layout(
                hovermode='x unified',  # Unify the hover information
                xaxis=dict(tickmode='linear')  # Ensure all x-axis labels are shown
            )
            st.plotly_chart(fig_time_of_day)

            st.write("""
            **Observation:** The plot shows how bike-sharing activity varies throughout the month, segmented by different times of the day.
            - Evening and Morning times see higher bike-sharing activity, suggesting that many bike users are likely commuters traveling to and from work.
            - Afternoon shows a moderate level of activity, likely due to leisure or errand trips.
            - Night has the lowest activity and fluctuates the least, indicating fewer bike-sharing trips during late hours and more consistent usage patterns at night.
            """)
        
        elif time_variable == "Weekday vs Weekend Patterns":
            # Create counts for weekends and add a new column for labeling
            weekend_counts = combined[combined['is_Weekend_Name'] == 'Weekend'].groupby('Hour_of_Day').size().reset_index(name='Number of Bike-Sharing')
            weekend_counts['is_Weekend_Name'] = 'Weekend'

            # Create counts for weekdays and add a new column for labeling
            weekday_counts = combined[combined['is_Weekend_Name'] == 'Weekday'].groupby('Hour_of_Day').size().reset_index(name='Number of Bike-Sharing')
            weekday_counts['is_Weekend_Name'] = 'Weekday'

            # Combine both weekend and weekday counts
            combined_counts = pd.concat([weekend_counts, weekday_counts])

            # Create line plot to show hourly changes for weekends and weekdays
            fig_hourly = px.line(
                combined_counts,
                x='Hour_of_Day',
                y='Number of Bike-Sharing',
                color='is_Weekend_Name', 
                title="Hourly Bike-Sharing: Weekday vs Weekend",
                labels={"Hour_of_Day": "Hour of Day", "Number of Bike-Sharing": "Count"},
                markers=True,
                color_discrete_map={'Weekend': 'green', 'Weekday': 'lightgreen'}  # Use green shades for the lines
            )

            # Update y-axis to show numbers in xx,xxx format
            fig_hourly.update_layout(
                yaxis=dict(
                    tickformat=','
                ),
                xaxis=dict(
                    range=[0, 24]  # Ensure x-axis starts from 0 and ends at 24
                )
            )
            fig_hourly.update_traces(mode='lines+markers', hovertemplate='N = %{y}')
            fig_hourly.update_layout(hovermode='x unified')
            fig_hourly.update_xaxes(tickmode='linear')  # Ensure all x-axis labels are shown
            st.plotly_chart(fig_hourly)

            st.write("""
                **Observation:** The plot illustrates the hourly bike-sharing patterns for weekdays and weekends.
                - **Weekdays** activity is generally higher than weekend activity, reflecting commuter use during the workweek.
                - On **weekdays**, there aretwo distinct peaks during the **morning** (commuter rush hour) and **early evening** (after-work period), indicating high bike-sharing activity during commute times.
                - **Weekends** show a more consistent distribution of bike-sharing throughout the day, with a slight increase in the afternoon.
                - Both weekends and weekdays tend to have low activity during late-night hours.
            """)

        elif time_variable == "Hourly Patterns by Day":            
            # Define the correct order for days of the week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            # Ensure 'Day_of_Week_Name' is treated as a categorical variable with this order
            combined['Day_of_Week_Name'] = pd.Categorical(combined['Day_of_Week_Name'], categories=day_order, ordered=True)

            # Group and unstack to create a 2D matrix
            avg_hour_day = combined.groupby(['Hour_of_Day', 'Day_of_Week_Name']).size().unstack(fill_value=0)

            # Reorder the columns to ensure the days of the week are in the correct order (optional, as pd.Categorical should handle this)
            avg_hour_day = avg_hour_day.reindex(columns=day_order)

            # Create a heatmap
            fig_avg_hour_day = px.imshow(
                avg_hour_day,
                aspect="auto",
                labels=dict(x="Day of Week", y="Hour of Day", color="Average Bike-Sharing"),
                title="Average Bike-Sharing by Hour and Day of Week",
                color_continuous_scale='YlGnBu',
                text_auto='.0f'  # Display annotations with 0 decimal places
            )

            # Ensure that x-axis follows the correct day order
            fig_avg_hour_day.update_layout(
                xaxis_title='Day of Week', 
                yaxis_title='Hour of Day',
                xaxis_categoryorder='array',  # Ensure days of the week are displayed in correct order
                xaxis_tickvals=[0, 1, 2, 3, 4, 5, 6],  # Tick values for the days
                xaxis_ticktext=day_order,  # Text for the tick values
                autosize=True,
                yaxis=dict(tickmode='linear')  # Ensure all y-axis labels are shown
            )

            # Increase the size of the heatmap
            fig_avg_hour_day.update_layout(
                height=800,  
                width=1200   
            )

            st.plotly_chart(fig_avg_hour_day)

            st.markdown("""
            **Observation:** The heatmap visualizes the average number of bike-sharing trips by the hour of the day across each day of the week, showing **clear peak times for bike-sharing** across the week.
            - **Weekdays**, especially **Tuesday and Thursdays**, show the highest activity, with a notable spike in the **late afternoon (4 PM - 7 PM)**. 
            - **Saturdays** consistently show the lowest activity.
            - These patterns highlight the influence of work schedules on bike-sharing trends, with significant usage concentrated during the weekday mornings and evenings.
                        """)

    if eda_option == "What impact does weather have on bike-sharing?":

        weather_option = st.selectbox ("Select one to visualize:", ["Weather Conditions", "Meteorological Factors"])

        if weather_option == "Weather Conditions":   

            st.subheader("Impact of Weather Conditions on Bike-Sharing")

            weather_counts = combined.groupby('Weather Description').size().reset_index(name='Number of Bike-Sharing')
            total_bike_sharing = weather_counts['Number of Bike-Sharing'].sum()
            weather_counts['Percentage'] = (weather_counts['Number of Bike-Sharing'] / total_bike_sharing) * 100

            # Create a bar plot for the number of bike-sharing trips by weather condition
            fig_weather_bar = px.bar(
                weather_counts,
                x='Weather Description',  
                y='Number of Bike-Sharing',
                color='Number of Bike-Sharing',
                color_continuous_scale='Viridis',
                title="Bike-Sharing Distribution by Weather Condition",
                labels={'Number of Bike-Sharing': 'Bike-Sharing Trip Count'}  # Update the color legend
            )

            # Format y-axis numbers as x,xxx
            fig_weather_bar.update_layout(yaxis=dict(tickformat=','))

            # Annotate bars with counts and percentages
            fig_weather_bar.update_traces(
                texttemplate='%{y:,} (%{customdata[0]:.2f}%)',  # Format the text with commas and percentages
                textposition='outside',  # Position the text outside the bars
                customdata=weather_counts[['Percentage']].values  # Add percentage data for annotation
            )

            # Hide the legend and y-axis
            fig_weather_bar.update_layout(showlegend=False, yaxis=dict(visible=False))

            # Adjust the height of the plot
            fig_weather_bar.update_layout(height=500)

            st.plotly_chart(fig_weather_bar)

            st.markdown("""
            **Observation:**
            - **Partly Cloudy** and **Clear Sky** weather conditions have the highest number of bike-sharing trips.
            - Wet weather conditions, such as **Heavy Drizzle** and **Moderate Rain**, see the lowest number of bike-sharing trips.
                        """)
            
            st.divider()

            # Add radio buttons for heatmap selection
            weather_condition_option = st.radio(
                "Select a heatmap to display:",
                ["Hourly Bike-Sharing by Weather Condition", "Bike-Sharing by Day of Week and Weather Condition"]
            )

            if weather_condition_option == "Bike-Sharing Trips per Hour by Weather Condition":
                # Group and unstack to create a 2D matrix for average bike-sharing by hour and weather condition
                avg_hour_weather = combined.groupby(['Hour_of_Day', 'Weather Description']).size().unstack(fill_value=0)

                fig_avg_hour_weather = px.imshow(
                    avg_hour_weather,
                    aspect="auto",
                    labels=dict(y="Hour of Day", x="Weather Condition", color="Bike-Sharing Trip Count"),
                    title="Bike-Sharing Trips per Hour by Weather Condition",
                    color_continuous_scale='Blues',  
                    text_auto='.0f'  # Display annotations with 0 decimal places
                )
                fig_avg_hour_weather.update_layout(
                    height=800,  
                    width=1200,
                    xaxis_title=None, # Remove x-axis label
                    yaxis_title='Hour of Day',  
                    yaxis=dict(tickmode='linear', tick0=0, dtick=1)  # Ensure all y-axis labels are shown from 0 to 23
                )

                st.plotly_chart(fig_avg_hour_weather)

                st.markdown("""
                            **Observation:** This heatmap visualizes the average number of bike-sharing trips across different hours of the day, broken down by various weather conditions.
                            - **Clear Sky** and **Partly Cloudy** conditions consistently show high bike-sharing activity, particularly during the evening hours (5 PM - 7 PM).
                            - **Wet Weather**, such as Drizzle, Heavy Drizzle, and Moderate Rain conditions, exhibit significantly lower bike-sharing activity throughout the day.
                            - The highest activity occurs around 6 PM, especially under **Clear Sky** weather conditions.
                            """)

            if weather_condition_option == "Bike-Sharing by Day of Week and Weather Condition":

                # Group and unstack to create a 2D matrix for average bike-sharing by day and weather condition
                avg_day_weather = combined.groupby(['Weather Description', 'Day_of_Week']).size().unstack(fill_value=0)

                # Define the correct order for days of the week
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

                # Ensure 'Day_of_Week' is treated as a categorical variable with this order
                avg_day_weather.columns = pd.Categorical(avg_day_weather.columns, categories=range(1, 8), ordered=True)
                avg_day_weather.columns = [day_order[day - 1] for day in avg_day_weather.columns]

                # Heatmap for average bike-sharing by day and weather condition
                fig_avg_day_weather = px.imshow(
                    avg_day_weather,
                    aspect="auto",
                    labels=dict(y="Weather Condition", x="Day of Week", color="Bike-Sharing Trip Count"),
                    title="Bike-Sharing by Day of Week and Weather Condition",
                    color_continuous_scale='Blues',
                    text_auto='.0f'
                )
                fig_avg_day_weather.update_layout(
                    xaxis_title=None, 
                    yaxis_title=None
                )
                st.plotly_chart(fig_avg_day_weather)

                st.markdown("""
                             **Observation:** The heatmap visualizes the average number of bike-sharing trips on each day of the week, categorized by various weather conditions.
                            - **Wednesday** and **Thursday** show the highest bike-sharing activity, especially under **Partly Cloudy** and **Clear Sky** conditions.
                            - **Overcast** and **Cloudy** conditions still see moderate activity, particularly in the middle of the week.
                            - **Heavy Drizzle** and **Moderate Rain** conditions result in minimal bike-sharing, with nearly no trips observed on certain days.
                            """)

        if weather_option == "Meteorological Factors":

            st.subheader("Impact of Key Meterological Factors on Bike-Sharing")

            # Violin Plot - Distribution of Bike-Sharing by Weather Variables
            # Reshape the weather columns into long format
            weather_cols = ['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']

            # Convert the wide data to long format
            long_format = pd.melt(combined, id_vars=[], value_vars=weather_cols, var_name='Weather_Variable', value_name='Value')

            # Add meaningful labels for each weather variable
            weather_variable_labels = {
                'real_temperature_C': 'Temperature (°C)',
                'humidity_percentage': 'Humidity (%)',
                'feels_like_temperature_C': 'Feels Like (°C)',
                'wind_speed_10m': 'Wind Speed (m/s)'
            }

            # Map the weather variables to the labels
            long_format['Weather_Variable'] = long_format['Weather_Variable'].map(weather_variable_labels)

            # Violin plot for weather variables
            fig_weather_facet = px.violin(
                long_format,
                y='Value',
                x='Weather_Variable',
                box=True,  # Add box plot inside the violin
                color='Weather_Variable',
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            # Update layout
            fig_weather_facet.update_layout(
                title="Bike-Sharing Distribution by Temperature, Humidity, Feels Like Temperature, and Wind Speed",
                xaxis_title=None,  # Remove x-axis title
                yaxis_title="Value (Normalized)",  
                yaxis_range=[0, 1],  # Normalize the y-axis range
                showlegend=False  # Hide the legend
            )

            # Update hover template to show 4 decimal places
            fig_weather_facet.update_traces(hovertemplate='%{y:.4f}')

            st.plotly_chart(fig_weather_facet)

            st.markdown("""
                        **Observation:** The violin plot shows the distribution of bike-sharing activity across different weather variables.
                        - Bike-sharing tends to occur more frequently in **moderate weather conditions**,
                        including mild temperatures, moderate humidity, and calm winds. 
                        - Extreme conditions, such as high humidity or strong winds, see less activity.
                        """)

            st.divider()

            # Linear Regression - Bike-Sharing vs. Weather Variables
            st.subheader("Impact of Weather on Bike-Sharing: Linear Regression Analysis")

            # Prepare the dataset for regression
            bike_weather_data = combined[['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']]

            # Add a column for Bike-Sharing counts (each row = 1 bike-sharing event)
            bike_weather_data['Bike-Sharing Count'] = 1

            # Group by weather variables
            bike_weather_grouped = bike_weather_data.groupby(['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']).size().reset_index(name='Bike-Sharing Count')

            # Select variable for regression
            selected_variable = st.selectbox(
                "Select a variable for linear regression:",
                options=['real_temperature_C', 'humidity_percentage', 'feels_like_temperature_C', 'wind_speed_10m']
            )

            if selected_variable:
                # Prepare data for regression
                X = bike_weather_grouped[[selected_variable]]  
                y = bike_weather_grouped['Bike-Sharing Count']  # Dependent variable: Bike-Sharing Count

                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Create and fit the linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)

                # Evaluate the model (R-squared, RMSE)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Get the coefficients for the regression equation
                slope = model.coef_[0]
                intercept = model.intercept_

                # Plot the regression line and the data using Plotly
                fig_regression = px.scatter(
                    x=X_test[selected_variable], 
                    y=y_test, 
                    labels={'x': weather_variable_labels[selected_variable], 'y': 'Count'},
                    title=f'Bike-Sharing vs. {weather_variable_labels[selected_variable]} (Linear Regression)',
                    color_discrete_sequence=['#1f77b4']  # Better color for scatter points
                )

                # Add the regression line
                fig_regression.add_trace(
                    go.Scatter(
                        x=X_test[selected_variable], 
                        y=y_pred, 
                        mode='lines',
                        line=dict(color='#ff7f0e'),  # Better color for regression line
                        name='Regression Line'
                    )
                )

                # Update layout
                fig_regression.update_layout(
                    xaxis_title=weather_variable_labels[selected_variable],
                    yaxis_title='Count',
                    showlegend=False
                )

                # Add R-squared and RMSE to the plot in separate locations
                fig_regression.add_annotation(
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    text=f"R-squared: {r2:.2f}<br>RMSE: {rmse:.2f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="#ffebcd",  # Changed background color for annotation
                    opacity=0.8
                )

                # Add annotation for the regression equation 
                fig_regression.add_annotation(
                    x=X_test[selected_variable].mean() + (X_test[selected_variable].max() - X_test[selected_variable].min()) * 0.3,  # Adjust x position to the right
                    y=y_pred.mean() + (y_pred.max() - y_pred.min()) * 0.1,  # Adjust y position to be above the line
                    text=f'y = {slope:.2f}x + {intercept:.2f}',
                    showarrow=False,
                    font=dict(size=12, color="#ff7f0e"),  # Better color for annotation text
                    align="center",
                    bordercolor="#ff7f0e",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="#fff3e0",  # Better background color for annotation
                    opacity=0.8
                )

                st.plotly_chart(fig_regression)

            with st.popover("Conclusion"):
                st.markdown("""
                            **Observation:** The linear regression analysis shows the relationship between bike-sharing and a selected meteorological factor.
                            - Temperature and Feels Like Temperature have a positive, though weak, relationship with bike-sharing, indicating that higher temperatures lead to more bike-sharing trips.
                            - Humidity appears to have a negative relationship with bike-sharing, with fewer trips at higher humidity levels.
                            - Wind speed seems to have little to no noticeable effect on the number of trips.
                            """)
               

    if eda_option == "Which stations are more popular?":
       
        # Station selection
        station_option = st.radio("Select a station type:", ["Start station", "End station"])

        # Determine unique stations
        unique_stations = combined[station_option].nunique()

        # Slider for selecting the number of busiest stations to display
        top_n = st.slider(f"Select the number of busiest {station_option.lower()}s to display:", min_value=1, max_value=unique_stations, value=10, step=1)

        # Identify the busiest stations based on the selected station type
        top_stations = combined[station_option].value_counts().head(top_n).index
        top_station_data = combined[combined[station_option].isin(top_stations)]

        # Create a bar plot for the busiest stations
        fig_station = px.bar(
            top_station_data.groupby(station_option).size().reset_index(name='Number of Bike-Sharing').sort_values(by='Number of Bike-Sharing', ascending=False),
            x='Number of Bike-Sharing',  # Count on the x-axis
            y=station_option,             # Station names on the y-axis
            title=f"The Busiest {top_n} {station_option}s",
            labels={station_option: f"{station_option} Name", "Number of Bike-Sharing": "Count"},
            color=station_option          # Color by station name
        )

        # Hide the legend
        fig_station.update_layout(showlegend=False)

        # Update layout for the color bar
        fig_station.update_layout(coloraxis_colorbar=dict(title='Count'))

        # Display the plot in Streamlit
        st.plotly_chart(fig_station)

        # Calculate the percentage of trips for the busiest stations
        total_trips = combined.shape[0]
        top_station_trips = top_station_data.shape[0]
        percentage = (top_station_trips / total_trips) * 100

        # Combine the explanation of the plot with the percentage calculation
        st.markdown(f"""
        This plot shows the **Number of Bike-Sharing Trips** (x-axis) for the **Busiest {top_n} {station_option.lower()}s** (y-axis).
        - The x-axis represents the total count of bike-sharing trips for each station.
        - The y-axis lists the station names.
        - Message: These busiest {station_option.lower()}s represent **{percentage:.2f}%** of the total bike-sharing trips.
        """)

        st.divider()

        # Detailed Analysis of the Busiest Stations
        st.subheader(f"The Busiest {top_n} {station_option}s: Detailed Analysis")

        # Create a bar plot for daily distribution
        daily_counts = top_station_data.groupby(['Day_of_Week_Name', station_option]).size().reset_index(name='Number of Bike-Sharing')
        fig_daily = px.bar(
            daily_counts,
            x='Day_of_Week_Name',
            y='Number of Bike-Sharing',
            color=station_option,
            title=f"Daily Distribution of Bike-Sharing for the Busiest {top_n} {station_option}s",
            barmode='group',
            category_orders={"Day_of_Week_Name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},  # Ensure correct order
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_daily.update_layout(
            dragmode='zoom', 
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1, traceorder='normal', title_text='Stations', itemsizing='constant', itemclick='toggleothers', itemdoubleclick='toggle'),  # Move legend to the right and hide until clicked
            xaxis_title=None,  # Remove x-axis label
            yaxis_title=None   # Remove y-axis label
        )
        st.plotly_chart(fig_daily)

        # Create a line plot for hourly distribution
        hourly_counts = top_station_data.groupby(['Hour_of_Day', station_option]).size().reset_index(name='Number of Bike-Sharing')
        fig_hourly = px.line(
            hourly_counts,
            x='Hour_of_Day',
            y='Number of Bike-Sharing',
            color=station_option,
            title=f"Hourly Distribution of Bike-Sharing for the Busiest {top_n} {station_option}s",
            labels={"Hour_of_Day": "Hour of Day", "Number of Bike-Sharing": "Count"},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_hourly.update_xaxes(tickmode='linear', range=[0, 23])  # Ensure all x-axis labels are shown and start from 0
        fig_hourly.update_layout(
            hovermode='x unified', 
            dragmode='zoom', 
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1, traceorder='normal', title_text='Stations', itemsizing='constant', itemclick='toggleothers', itemdoubleclick='toggle')  # Move legend to the right and hide until clicked
        )
        
        st.plotly_chart(fig_hourly)

        # Optional interactive plot for Day-of-Month distribution
        # def plot_day_of_month_distribution(data, station_option):
        #     fig_day_of_month = px.histogram(
        #         data,
        #         x='Day_of_Month',
        #         nbins=31,
        #         title=f"Day-of-Month Distribution for the Busiest {top_n} {station_option}s",
        #         labels={'Day_of_Month': 'Day of Month', 'count': 'Number of Bike-Sharing Trips'},
        #         color_discrete_sequence=['green' if station_option == "Start station" else 'blue'],
        #         opacity=0.7
        #     )
            
        #     fig_day_of_month.update_layout(
        #         hovermode="x unified",
        #         xaxis_title="Day of Month",
        #         yaxis_title="Number of Bike-Sharing Trips",
        #         plot_bgcolor='rgba(0,0,0,0)',
        #         paper_bgcolor='rgba(0,0,0,0)',
        #         xaxis_showgrid=False,
        #         yaxis_showgrid=False
        #     )
            
        #     return fig_day_of_month

        # Display Day-of-Month Distribution plot (optional, can be toggled on demand)
        # st.plotly_chart(plot_day_of_month_distribution(top_station_data, station_option))

        st.markdown("""
        **Messages:**

        - **Daily Distribution**: Shows bike-sharing activity across the week for the busiest stations, helping predict peak demand.
        - **Hourly Distribution**: Shows rush hours when bike-sharing spikes, useful for optimizing bike availability during peak times.
        """)

    if eda_option == "What do we know by now?":
        st.subheader("What we know?")
        
        cols1, cols2 = st.columns([1,2])

        with cols1:
            st.image('https://media.timeout.com/images/101651783/750/562/image.jpg', 
                    width=300)
        with cols2:
            st.markdown("""
                - **Temporal Patterns**: Weekdays show clear peaks in bike-sharing during the morning and evening rush hours, with midweek seeing the highest activity. 
                - **Weather Impact**: Favorable weather conditions, such as clear skies and mild temperatures, correlate with higher bike-sharing activity, while extreme conditions like rain or high humidity reduce usage.
                - **Weather Regression Analysis**: Temperature shows a weak positive correlation with bike-sharing counts, while humidity has a weak negative correlation. Wind speed appears to have minimal effect on usage.
                            """)
        st.divider()
        
        st.subheader("What Questions Could We Explore Next?")

        st.markdown("""
                - How can we **optimize station locations** and bike availability to better serve commercial and residential areas based on demand patterns?
                - Would bike-sharing usage be more accurately predicted through **time series forecasting** or **clustering analysis** of stations and users?
                - Could **station grouping** based on regions or routes help visualize cross-region commuter patterns using map-based tools?
                - How do **combinations of weather variables** (temperature, humidity, wind speed) collectively impact bike usage, and could **clustering weather conditions** provide clearer insights?
                - Is there a measurable **commuter effect** that can be confirmed through deeper analysis of weekday trips, peak times, and station locations?
        """)

        st.subheader("Next Steps")

        st.markdown("""
                - **Explore advanced forecasting** to predict bike-sharing demand during peak times and under varying weather conditions.
                - **Use clustering techniques** to uncover hidden patterns in station usage and weather effects on bike-sharing.
                - **Investigate commuter patterns** and identify routes with heavy traffic to optimize service for commuters.
        """)
