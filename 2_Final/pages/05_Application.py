#!/usr/bin/env python

import streamlit as st

st.set_page_config(
    page_title="Application",
    layout="wide"
)

st.markdown('''
## Make Predictions Live

The best model results arise from using the Kernal Ridge Regressor with the given default values.  
However, feel free to choose the model you'd like to make the prediction with:
''')

model_name = st.selectbox(
    'Select Model: ',
    ('Kernel Ridge', 'Neural Network', 'Linear Regression')
)

model = st.session_state[model_name]
categorical, numerical, target, ohe, sc = st.session_state['stuff']

weats = ['Cloudy', 'Light Rain', 'Heavy Rain']
seas = ['Spring', 'Summer', 'Fall', 'Winter']
boo = ['No', 'Yes']

df = {}

for cat in categorical:
    if cat == 'season':
        thing = seas.index(st.selectbox("Season:", seas)) + 1

    if cat == 'holiday':
        thing = boo.index(st.selectbox("Holiday:", boo))

    if cat == 'workingday':
        thing = boo.index(st.selectbox("Working Day:", boo))

    if cat == 'weather':
        thing = weats.index(st.selectbox("Weather:", weats)) + 1

    df[cat] = [thing]

for num in numerical:
    thing = st.slider(num.capitalize(), 0.0, 100.0, 0.1)  # Adjust the range as needed
    df[num] = [thing]

import pandas as pd

X = pd.DataFrame(df)

'### Data:'
st.write(X)

def transform(X, ohe, sc):
    # Dummy transform function, replace with actual implementation
    return X, None, None

X_trans, _, _ = transform(X, ohe, sc)

'### Transformed Data:'
st.write(X_trans)

try:
    st.markdown(f'''
    ### Prediction:
    
    Today, you will have {model.predict(X_trans)[0]:.0f} {target} users per hour.
    ''')
except:
    st.markdown('''
    #### --- Train the model, then try again ---
    ''')
