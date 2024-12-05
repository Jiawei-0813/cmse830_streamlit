import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Feature Engineering on Combined Dataset",
    layout="wide"
)

st.title("🛠️ Feature Engineering: Combined Dataset")

# Ensure the combined dataset is available
if "bike_data_cleaned" in st.session_state and "weather_data_cleaned" in st.session_state:
    bike_data = st.session_state["bike_data_cleaned"]
    weather_data = st.session_state["weather_data_cleaned"]

    # Combine the datasets
    combined = pd.merge(bike_data, weather_data, on="Date", how="inner")
    st.session_state["combined_data_raw"] = combined

    # =====================
    # Feature Engineering
    # =====================
    st.subheader("🛠️ Feature Engineering Options")
    
    # Time-Based Features
    if st.checkbox("Create Time-Based Features (e.g., Hour, Day, Weekend)"):
        combined['Hour_of_Day'] = combined['Date'].dt.hour
        combined['Day_of_Week'] = combined['Date'].dt.dayofweek + 1
        combined['is_Weekend'] = combined['Day_of_Week'].apply(lambda x: 1 if x > 5 else 0)
        combined['Time_of_Day'] = combined['Hour_of_Day'].apply(
            lambda x: 'Morning' if 5 <= x < 12 else 
                      'Afternoon' if 12 <= x < 17 else 
                      'Evening' if 17 <= x < 21 else 'Night'
        )
        st.success("Time-based features created!")

    # Interaction Features
    if st.checkbox("Create Interaction Features (e.g., Bike Count x Humidity)"):
        if 'Bike Counts' not in combined.columns:
            combined['Bike Counts'] = combined.groupby('Date')['Start station'].transform('count')
        combined['Bike_Humidity_Interaction'] = combined['Bike Counts'] * combined['humidity_percentage']
        st.success("Interaction features created!")

    # Scaling
    if st.checkbox("Apply Min-Max Scaling to Weather Features"):
        scaler = MinMaxScaler()
        weather_cols = ['real_temperature_C', 'feels_like_temperature_C', 'humidity_percentage', 'wind_speed_10m']
        for col in weather_cols:
            combined[f"{col}_scaled"] = scaler.fit_transform(combined[[col]])
        st.success("Scaling applied to weather features!")

    # Encoding Categorical Features
    if st.checkbox("Label Encode `Time_of_Day`"):
        combined['Time_of_Day_Encoded'] = LabelEncoder().fit_transform(combined['Time_of_Day'])
        st.success("Label Encoding applied to `Time_of_Day`!")

    # Display Engineered Data
    st.write("Preview of Engineered Features:")
    st.dataframe(combined.head())

    # Save the engineered dataset
    st.session_state["combined_data"] = combined
    st.success("Combined dataset with engineered features saved successfully!")
else:
    st.error("Please preprocess and combine the bike and weather datasets first!")