import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

# Load the dataset
hdf = pd.read_csv("C:\\Datasets\\heart.csv")

    # Prepare the data
x = hdf.drop(['target'], axis=1)
y = hdf['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

# Train the model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Predictions
ypred_train = dt.predict(x_train)
ypred_test = dt.predict(x_test)

# Metrics
train_accuracy = accuracy_score(y_train, ypred_train)
test_accuracy = accuracy_score(y_test, ypred_test)
    

st.title("Heart Disease Prediction")
    
    # Sidebar for user inputs
st.sidebar.header("Enter Patient Details")
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=90, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise (0-2)", [0, 1, 2])
ca = st.sidebar.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    # Make prediction
if st.sidebar.button("Predict"):

    sample_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = dt.predict(sample_input)

    if prediction[0] == 1:

        st.sidebar.success("The model predicts  Patient has: **Heart Disease**.")
    else:

        st.sidebar.success("The model predicts Patient has:**No Heart Disease**.")


# Display metrics
st.subheader("Model Performance")
st.write(f"**Training Accuracy:** {train_accuracy:.2f}")
st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
st.subheader("Classification Report")
st.text(confusion_matrix(y_test, ypred_test))
