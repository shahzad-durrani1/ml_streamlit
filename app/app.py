import streamlit as st
import pandas as pd
from joblib import load
import dill
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pretrained model
with open('model/employee_pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('model/employee_feature_dict.pkl')

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Churn Prediction App", page_icon="📊", layout="wide")

# App Title
st.title('📊 Employee Churn Prediction App')
st.markdown("### A sleek and interactive application to predict employee churn based on key metrics.")

# Create sidebar for user input
st.sidebar.header('Input Features')

# Display categorical features
st.sidebar.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}
for i, col in enumerate(categorical_input.get('Column Name')):
    categorical_input_vals[col] = st.sidebar.selectbox(f"{col}", categorical_input.get('Members')[i])

# Display numerical features
st.sidebar.subheader('Numerical Features')
numerical_input_vals = {}
numerical_input_vals['Age'] = st.sidebar.slider('Age', min_value=0, max_value=100, value=50)
numerical_input_vals['ExperienceInCurrentDomain'] = st.sidebar.slider('Experience in Current Domain', min_value=0, max_value=25, value=5)
numerical_input_vals['JoiningYear'] = st.sidebar.slider('Joining Year', min_value=2010, max_value=2030, value=2020)

# Combine numerical and categorical input dicts
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame.from_dict(input_data, orient='index').T

# Main content section
st.write("### Employee Information")
st.dataframe(input_data)

# Churn Prediction
if st.button('Predict Churn'):
    prediction = predict_churn(input_data)[0]
    translation_dict = {1: "Expected", 0: "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    st.write('### Prediction Result')
    if prediction == 1:
        st.error(f'The model predicts: **Employee is {prediction_translate} to leave** 😟')
    else:
        st.success(f'The model predicts: **Employee is {prediction_translate} to stay** 😊')


# Add footer information
st.markdown("---")
st.markdown("#### 📌 Disclaimer: This prediction is based on historical data and should not be the sole factor in decision making.")
st.markdown("#### 🛠️ Developed by [Your Name] - A Data Science Enthusiast")
