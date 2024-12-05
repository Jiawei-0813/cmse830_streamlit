import streamlit as st

st.set_page_config(
    page_title="Conclusion",
    layout="wide"
)

st.subheader("What do we know?")

st.image('https://media.timeout.com/images/101651783/750/562/image.jpg')
st.markdown("""
    - **Temporal Patterns**: Weekdays show clear peaks in bike-sharing during the morning and evening rush hours, with midweek seeing the highest activity. 
    - **Weather Impact**: Favorable weather conditions, such as clear skies and mild temperatures, correlate with higher bike-sharing activity, while extreme conditions like rain or high humidity reduce usage.
    - **Weather Regression Analysis**: Temperature shows a weak positive correlation with bike-sharing counts, while humidity has a weak negative correlation. Wind speed appears to have minimal effect on usage.
""")
st.divider()

