#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from prophet import Prophet
import pickle

# Load the saved Prophet model
with open("prophet_model.pkl", 'rb') as f:
    model = pickle.load(f)
    print(model)
    


# In[2]:


# Function to make predictions for a given date
def make_predictions(selected_date):
    future_df = pd.DataFrame({'ds': [selected_date]})
    forecast = model.predict(future_df)
    return forecast['yhat'].values[0]

# Streamlit interface
st.title('Bitcoin Price Predictor')
selected_date = st.date_input('Select a date:', min_value=pd.to_datetime('today'))

if st.button('Predict'):
    prediction = make_predictions(selected_date)
    st.write('Predicted value for {}: {}'.format(selected_date, prediction))


# In[ ]:




