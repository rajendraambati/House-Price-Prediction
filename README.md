# House-Price-Prediction

## *Live Link : https://house-price-prediction-rajendraambati.streamlit.app/*

![house_price_predictino home](https://github.com/user-attachments/assets/6612e9b6-586c-4ddd-8f8a-c95aa5766bdf)
![house_price_predictino prediction](https://github.com/user-attachments/assets/87d6ce1b-645b-4bb7-aeea-aa97d4d382d4)

Creating a house price prediction model and deploying it using Streamlit can be a rewarding project that showcases the power of machine learning in real estate. In this blog, I will guide you through the process of building this application, from data preparation to deployment.

# Project Overview
The goal of this project is to predict house prices based on various features such as location, size, and demographics. We will use a dataset containing housing information from California and employ a linear regression model for predictions. Finally, we will deploy the application using Streamlit, making it accessible via a web interface.

# Key Technologies Used
  - Python: The primary programming language for data manipulation and model training.
  - Pandas: For data handling and preprocessing.
  - Scikit-learn: To build and evaluate the machine learning model.
  - Streamlit: For creating the web application interface.
  - Pickle: To save and load the trained model.
# Implementation Steps
## Setting Up the Environment
First, ensure you have Python installed along with the necessary libraries. You can create a virtual environment and install the required packages with:

```py
pip install pandas scikit-learn streamlit
```

## 2. Data Preparation
We will use the California housing dataset, which contains various features that influence house prices. Here’s how to load and explore the data:
python

```py
import pandas as pd
import pandas as pd

# Load dataset
data = pd.read_csv('california_housing_train.csv')

# Display basic information about the dataset
print(data.head())
print(data.info())
#Load dataset
data = pd.read_csv('california_housing_train.csv')

# Display basic information about the dataset
print(data.head())
print(data.info())
```

## 3. Data Exploration and Visualization
Before training our model, it's essential to understand our data. We can visualize distributions and box plots to identify any anomalies:
```py
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing distributions of features
for column in data.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()
```

## 4. Model Training
Next, we will split the dataset into training and testing sets, train a linear regression model, and evaluate its performance.

```py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Split data into features and target variable
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Save the trained model using pickle
with open('finalized_model.sav', 'wb') as file:
    pickle.dump(model, file)
```

## 5. Building the Streamlit Application
Now we will create a simple Streamlit application to allow users to input values for prediction.

```py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('finalized_model.sav', 'rb'))

# Streamlit app title
st.title('California Housing Price Prediction')

# Input features using sliders
longitude = st.sidebar.slider('Longitude', -124.35, -114.31)
latitude = st.sidebar.slider('Latitude', 32.54, 41.95)
housing_median_age = st.sidebar.slider('Housing Median Age', 1, 52)
total_rooms = st.sidebar.slider('Total Rooms', 1, 37937)
total_bedrooms = st.sidebar.slider('Total Bedrooms', 1, 6445)
population = st.sidebar.slider('Population', 3, 35682)
households = st.sidebar.slider('Households', 1, 6082)
median_income = st.sidebar.slider('Median Income', 0.4999, 15.0001)

# Create a DataFrame from user input
input_features = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_features)
    st.write(f'Predicted House Price: ${prediction[0]:,.2f}')
```

## 6. Running the Application
To run your Streamlit application:
  - Save your code in a Python file (e.g., app.py).
  - Open your terminal or command prompt.
  - Navigate to your project directory.
  - Run the following command:

```py
streamlit run app.py
```

This command will launch your web browser with the Streamlit interface where you can input values and see predictions.

# Conclusion

In this project, we built a house price prediction model using linear regression and deployed it using Streamlit for easy access through a web interface. This application not only demonstrates machine learning concepts but also provides a practical tool for estimating real estate values based on various features.

Feel free to enhance this project by experimenting with different models or adding more features to improve accuracy! You can find the complete code on my [GitHub repository](https://github.com/rajendraambati/House-Price-Prediction) for further exploration or collaboration.
