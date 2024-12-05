#!/usr/bin/env python

import streamlit as st

st.set_page_config(
    page_title="Preprocessing",
    layout="wide"
)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
from data_loader import load_bike_data, load_weather_data

# Ensure the data is loaded into session_state
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

st.header('Data Preprocessing & Feature Engineering')

# Create tabs for datasets
bike_tab, weather_tab, combined_tab = st.tabs(["🚴 Bike Dataset", "🌤️ Weather Dataset", "🔍 Combined Insights"])

with bike_tab:
    bike_0 = st.session_state["bike_data_raw"]

    st.subheader('Data Cleaning')
    bike_1 = bike_0.copy() # Create a copy of the original dataset

    # Convert 'Start date' and 'End date' to datetime format
    bike_1['Start date'] = pd.to_datetime(bike_1['Start date'], errors='coerce')
    bike_1['End date'] = pd.to_datetime(bike_1['End date'], errors='coerce')

    # Extract `Date` from 'Start date'
    bike_1['Date'] = bike_1['Start date'].dt.floor('T') # Round down to the nearest minute

    # Convert 'Total duration (ms)' from milliseconds to minutes
    bike_1['Total duration (m)'] = round(bike_1['Total duration (ms)'] / 60000, 0)

    st.write(bike_1[['Start date', 'End date', 'Date', 'Total duration (m)']].dtypes.apply(lambda x: x.name).to_frame('Type (Corrected)').style.set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}]
    ))

    # Drop redundant/irrelevant columns
    bike_1.drop(columns=['Number', 'Start date', 'End date', 'Total duration', 'Total duration (ms)', 'Bike number'], inplace=True)

    st.write(""" 
    - `Date` was extracted from `Start date` in `yyyy-mm-dd HH:MM` format for merging with the weather dataset.
    - `Total duration (m)` was converted to minutes.
    - Redundant columns were dropped after confirming the changes.
    """)
    
    st.divider()

    st.subheader('Feature Engineering - Encoding')    
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

        col1, col2 = st.columns([1,1])
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
            with st.popover("Deferred at this stage."):
                st.write(
                    """While one-hot encoding can be beneficial for machine learning models, the large number of unique stations complicates the dataset and could lead to the curse of dimensionality. Although we could focus on the top 10 or 20 stations, it may be more effective to postpone this step until after EDA and feature selection. Since each station name has a consistent station number, we might find that encoding isn’t necessary."""
                )

    st.divider ()

    st.subheader('Transformation & Outlier Detection')
    # Check for outliers in 'Total duration (m)'
    st.write(bike_1['Total duration (m)'].describe().to_frame().T)
    
    st.write('**Initial Observations:**')
    st.write('- Some trips lasting over 19,961 minutes (more than 13 days).')
    st.write('- The standard deviation (218.02 minutes) is significantly higher than the mean (25.43 minutes), indicating the data is widely spread out, with some trips having durations significantly longer than the average.')

    # Check for zero values
    zero_count = (bike_1['Total duration (m)'] == 0).sum()
    zero_percent = round((zero_count / len(bike_1)) * 100, 2)
    st.write(f"- There are {zero_count:,} ({zero_percent}%) trips with a duration of less than 1 minute (specifically those under 30 seconds). These trips are likely errors or anomalies and should be removed.")
    bike_1 = bike_1[bike_1['Total duration (m)'] > 0] # Remove trips with duration less than 1 minute
    
    # Save updated bike data to session state
    st.session_state["bike_data_preprocessing"] = bike_1

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

    # Save updated bike data to session state
    st.session_state["bike_data_outlierremoved"] = bike_2

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

    # Save Cleaned Bike Data
    if "bike_data_cleaned" not in st.session_state:
        st.session_state["bike_data_cleaned"] = bike_3
    st.success("Bike data cleaned and saved!")

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

with weather_tab:
    weather_0 = st.session_state["weather_data_raw"]

    st.subheader('Data Cleaning')

    weather_1 = weather_0.copy()

    # Convert 'date' to datetime and remove timezone information
    try:
        weather_1['date'] = pd.to_datetime(weather_1['date'], errors='coerce')
        weather_1['date'] = weather_1['date'].dt.tz_localize(None)  # Remove timezone information
        weather_1['Date'] = weather_1['date'].dt.floor('T') # Round down to the nearest minute

        # st.write(weather_1['Date'].dtypes)
        # st.write(weather_1[['date', 'Date']].head())

        st.write(f"- `Date` was extracted in the consistent `yyyy-mm-dd HH:MM` format for merging.")
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

    # Drop redundant columns
    weather_1.drop(columns=['date'], inplace=True)

    st.divider()

    st.subheader('Transformation & Outlier Detection')
    
    weather_2 = weather_1.copy()

    numeric_columns = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'relative_humidity_2m']
    st.write(weather_2[numeric_columns].describe().T. round(2))

    st.write("Outliers detected using the IQR method:")
    # Define detect_outliers_iqr function
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Preprocessing
    for variable in numeric_columns:
        if variable == 'relative_humidity_2m':
            st.write(f"`{variable}` is already in the reasonable range [0, 100].")
        else:
            weather_2 = detect_outliers_iqr(weather_2, variable)
            st.write(f"- `{variable}`: Outliers removed.")

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

    # Save Cleaned Weather Data
    if "weather_data_cleaned" not in st.session_state:
        st.session_state["weather_data_cleaned"] = weather_2
    st.success("Weather data cleaned and saved!")

# Combined Tab
with combined_tab:
    if "bike_data_cleaned" in st.session_state and "weather_data_cleaned" in st.session_state:
        bike_3 = st.session_state["bike_data_cleaned"]
        weather_2 = st.session_state["weather_data_cleaned"]
        st.markdown("""
        Combining the cleaned bike-sharing and weather datasets to create a holistic view. 
        This integration includes time-related and weather-based features, 
        setting the stage for exploratory data analysis (EDA) and predictive modeling. 
        Here's a preview:
        """)

        combined = pd.merge(bike_3, weather_2, on="Date", how="inner")
        st.session_state["combined_data"] = combined
        st.success("The datasets have been successfully merged based on the common `Date` column.")
    else:
        st.error("Cleaned datasets are not available in session state. Please preprocess the data first.")
        st.stop()

    # Time-Based Features Creation
    st.subheader("Feature Engineering: Time-Based Features Creation")
    st.markdown("""
                We extract time-based features from the `Date` column to provide additional context for analysis.
                """)

    combined['Day_of_Month'] = combined['Date'].dt.day  # 1-31
    combined['Day_of_Week'] = combined['Date'].dt.dayofweek + 1  # Monday=1, Sunday=7
    combined['Hour_of_Day'] = combined['Date'].dt.hour  # 0-23
    combined['is_Weekend'] = combined['Day_of_Week'].apply(lambda x: 1 if x > 5 else 0) # 1=Weekend, 0=Weekday
    combined['Time_of_Day'] = combined['Hour_of_Day'].apply(
        lambda hour: 'Morning' if 5 <= hour < 12 else
                    'Afternoon' if 12 <= hour < 17 else
                    'Evening' if 17 <= hour < 21 else 'Night'
                    )
        # Encoding Categorical Features
    if st.checkbox("Label Encode `Time_of_Day`"):
        combined['Time_of_Day_Encoded'] = LabelEncoder().fit_transform(combined['Time_of_Day'])
        st.success("Label Encoding applied to `Time_of_Day`!")

    # Mappings for readability
    day_of_week_mapping = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
    combined['Day_of_Week_Name'] = combined['Day_of_Week'].map(day_of_week_mapping)
    combined['is_Weekend_Name'] = combined['is_Weekend'].map({0: "Weekday", 1: "Weekend"})

    # Preview of newly created features
    validation_samples = combined[['Date', 'Day_of_Month', 'Day_of_Week_Name', 'Hour_of_Day', 'is_Weekend_Name', 'Time_of_Day']].sample(5)
    st.write("Sample rows for validation:")
    st.dataframe(validation_samples)

    # Interactive feature selection for distribution visualization
    selected_feature = st.selectbox(
        "Select a feature to visualize its distribution:",
        validation_samples.columns[1:]  # Exclude 'Date' from the options
    )

    if selected_feature:
        # Calculate distribution
        feature_dist = combined[selected_feature].value_counts().sort_index()

        # Create an interactive Plotly bar chart
        fig = px.bar(
            feature_dist,
            x=feature_dist.index,
            y=feature_dist.values,
            title=f"Distribution of `{selected_feature}`",
            labels={"x": selected_feature, "y": "Count"},
            text=feature_dist.values,  # Add counts as text annotations
            template="plotly_white",  # Use a clean template
        )

        # Adjust layout for better aesthetics
        fig.update_layout(
            xaxis=dict(title=selected_feature, tickmode='linear', tickangle=45),  # Rotate x-ticks if necessary
            yaxis=dict(title="Count"),
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
        )

        st.plotly_chart(fig)

    # Document the features created
    st.markdown("""
    **New Features Created:**
    1. **`Day_of_Month`:** Extracted day of the month (1-31).
    2. **`Day_of_Week`:** Day of the week as a number (Monday=1, Sunday=7).
    3. **`Day_of_Week_Name`:** Day of the week as a categorical label.
    4. **`Hour_of_Day`:** Hour of the day (0-23).
    5. **`is_Weekend`:** Binary indicator for weekends (1=Weekend, 0=Weekday).
    6. **`is_Weekend_Name`:** Readable labels for weekend/weekday.
    7. **`Time_of_Day`:** Categorical labels for time of day (Morning, Afternoon, Evening, Night).
    """)

    st.markdown(
        """
        <hr style="border: 2px solid #FF69B4;">
        """,
        unsafe_allow_html=True
    )

    # Renaming Features
    st.subheader("Renamed Columns")

    rename_map = {
        "Total duration (m)_log": "Total_duration_min",
        "temperature_2m_n": "real_temperature_C",
        "apparent_temperature_n": "feels_like_temperature_C",
        "wind_speed_10m_n": "wind_speed_10m",
        "relative_humidity_2m_n": "humidity_percentage",
    }
    combined.rename(columns=rename_map, inplace=True)
    
    # Display summary of renamed columns
    st.markdown(
        """
        During scaling and data cleaning, new variables were created and renamed for clarity:
        - `Total duration (m)_log` → `Total_duration_min`
        - `temperature_2m_n` → `real_temperature_C`
        - `apparent_temperature_n` → `feels_like_temperature_C`
        - `wind_speed_10m_n` → `wind_speed_10m`
        - `relative_humidity_2m_n` → `humidity_percentage`
        """
    )
    
    st.markdown(
        """
        <hr style="border: 2px solid #FF69B4;">
        """,
        unsafe_allow_html=True
    )

    # Add a button to trigger PCA
    # if st.checkbox("Apply PCA"):
    #     from sklearn.decomposition import PCA
    #     from sklearn.preprocessing import StandardScaler

    #     # Assuming `combined` is your DataFrame and you want to apply PCA on numerical features
    #     numerical_features = combined.select_dtypes(include=['float64', 'int64']).columns

    #     # Standardize the numerical features
    #     scaler = StandardScaler()
    #     combined_scaled = scaler.fit_transform(combined[numerical_features])

    #     # Apply PCA with 2 components as an example
    #     pca = PCA(n_components=2)
    #     combined_pca = pca.fit_transform(combined_scaled)

    #     # Visualize the explained variance to understand the contribution of each principal component
    #     explained_variance = pca.explained_variance_ratio_
    #     fig = px.bar(x=['PC1', 'PC2'], y=explained_variance, labels={'x': 'Principal Components', 'y': 'Explained Variance Ratio'},
    #                  title="Explained Variance Ratio by Principal Components")
    #     fig.update_layout(xaxis_title="Principal Components", yaxis_title="Explained Variance Ratio")
    #     st.plotly_chart(fig)

    #     # Optionally, print the explained variance ratio to directly see the values
    #     st.write("Explained Variance Ratio:", explained_variance)

    st.divider()
    
    st.subheader('**Analysis-Ready Data Overview**')
    st.dataframe(combined)

    cols1, cols2 = st.columns([1, 1])
    with cols1:
        if st.checkbox ("Summary Statistics"):
            st.markdown(f"""
                - **{combined.shape[0]:,}** rows and **{combined.shape[1]}** columns
                """)
            numeric_columns = ['Total_duration_min', 'real_temperature_C', 'feels_like_temperature_C', 'wind_speed_10m', 'humidity_percentage']
            st.write(combined[numeric_columns].describe().T.round(2))
    with cols2:
        if st.checkbox ("Show Data Types"):
            st.write (combined.dtypes.apply(lambda x: x.name).to_frame('Type').style.set_table_styles(
                    [{'selector': 'th', 'props': [('text-align', 'left')]},
                    {'selector': 'td', 'props': [('text-align', 'left')]}]
                ).set_table_attributes('style="width: auto;"'))

    # Save the dataset to a file (done in preprocessing step)
    combined.to_csv("data/1_london_bike_weather_2023.csv", index=False)

    # Save the combined dataset to session state
    if "combined_data" not in st.session_state:
        st.session_state["combined_data"] = combined
    st.success("The combined dataset has been saved successfully.")
    