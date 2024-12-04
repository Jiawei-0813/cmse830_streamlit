import streamlit as st

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
            use_container_width=True)