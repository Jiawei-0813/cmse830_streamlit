import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_bike_data():
    """Load the bike-sharing data."""
    path = "data/0_LondonBikeJourneyAug2023_small.csv"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_weather_data():
    """Load the weather data."""
    path = "data/0_london_weather_2023.csv"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    return pd.read_csv(path)