import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load the dataset
sdf = pd.read_csv("C:\\Datasets\\Advertising.csv")

# Drop the unnecessary index column
sdf = sdf.drop(columns=["Unnamed: 0"])

x = sdf[["TV", "Radio", "Newspaper"]]
y = sdf["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)


model = LinearRegression()
model.fit(x_train, y_train)


st.title("Sales Prediction using Multiple Linear Regression")

st.sidebar.header("Enter Advertising Budget")
tv_budget = st.sidebar.number_input("TV Budget", min_value=0, value=100)
radio_budget = st.sidebar.number_input("Radio Budget", min_value=0, value=100)
newspaper_budget = st.sidebar.number_input("Newspaper Budget", min_value=0, value=100)


if st.sidebar.button("Predict Sales"):
    sample_input = np.array([[tv_budget, radio_budget, newspaper_budget]])
    predicted_value = model.predict(sample_input)
    st.sidebar.success(f"Predicted Sales: {predicted_value[0]:.2f}")


