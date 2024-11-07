# prompt: correct this code for deploying the model using streamlit 'import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64
# @st.cache
# def get_fvalue(val):
#     feature_dict = {"No": 1, "Yes": 2}
#     return feature_dict[val]
# def get_value(val, my_dict):
#     return my_dict[val]
# if app_mode == 'Home':
#     st.title('Loan Prediction')
#     st.image('pexels-pixabay-280229.jpg')
#     st.markdown('Dataset:')
#     data = pd.read_csv('california_housing_train.csv')
#     st.write(data.head())
#     st.bar_chart(data[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
#        'total_bedrooms', 'population', 'households', 'median_income',
#        'median_house_value']].head(20))
# if app_mode == 'Prediction':
#     longitude = st.sidebar.slider('longitude', -124.35, -114.31)
#     latitude = st.sidebar.slider('latitude', -124.35, -114.31)
#     housing_median_age = st.sidebar.slider('housing_median_age', -124.35, -114.31)
#     total_rooms = st.sidebar.slider('total_rooms',

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Load the trained model
model = pickle.load(open('finalized_model.sav', 'rb'))

# Load data (replace with your actual data loading)
data = pd.read_csv('california_housing_train.csv')

# Define min and max values for sliders (replace with actual min/max from your data)
min_max_values = {}
for col in data.columns[:-1]:  # Exclude the target variable
    min_max_values[col] = (data[col].min(), data[col].max())

# Streamlit app
st.title('House Price Prediction') # Consider renaming to California Housing Price Prediction

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])

if app_mode == 'Home':
    st.title('House Price Prediction') # Consider renaming to California Housing Price Prediction
    st.image('pexels-pixabay-280229.jpg') # Make sure this image file exists
    st.markdown('Dataset:')
    st.write(data.head())
    st.bar_chart(data[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']].head(20))


if app_mode == 'Prediction':
    st.title('Prediction') # More descriptive title

    # Input features using sliders
    input_features = {}
    for col in data.columns[:-1]:
      input_features[col] = st.sidebar.slider(col, min_max_values[col][0], min_max_values[col][1])

    # Create a DataFrame from user input
    input_df = pd.DataFrame([input_features])

    # Make prediction
    prediction = model.predict(input_df)

    # Display the prediction
    st.write('Predicted Median House Value:', prediction[0])