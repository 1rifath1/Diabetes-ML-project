import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('E:/diabetes.csv')

# Streamlit app title
st.title('Diabetes Checkup')

# Display data description
st.subheader('Data')
st.write(df.describe())

# Visualize the data
st.title('Visualize')
st.bar_chart(df)

# Prepare the data for training
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define user report function
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.slider('Glucose', 0, 200, 110)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 20.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.5)
    age = st.sidebar.slider('Age', 21, 81, 33)
    
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get user input
user_data = user_report()

# Display user input
st.subheader('User Input:')
st.write(user_data)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions for the test set
y_predict = model.predict(x_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_predict)

# Display mean squared error
st.subheader('Mean Squared Error:')
st.write(mse)

# Make predictions for user input
user_result = model.predict(user_data)

# Display prediction result
st.subheader('Prediction:')
output = 'Healthy' if user_result[0] < 0.5 else 'Not Healthy'
st.write(output)
