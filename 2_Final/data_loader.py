import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_bike_data():
    """Load the bike-sharing data."""
    path = os.path.join(os.getcwd(), "data/0_LondonBikeJourneyAug2023_small.csv")
    st.write("Checking file path for bike data:", path)  # Debugging line
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    try:
        data = pd.read_csv(path)
        st.write("Bike data loaded successfully. Shape:", data.shape)  # Debugging line
        return data
    except Exception as e:
        st.error(f"Error loading bike data: {e}")
        st.stop()

@st.cache_data
def load_weather_data():
    """Load the weather data."""
    path = os.path.join(os.getcwd(), "data/0_london_weather_2023.csv")
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    return pd.read_csv(path)
st.write("Current Working Directory:", os.getcwd())