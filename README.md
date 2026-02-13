# Linear Regression & Multiple Regression
📌 Project Overview

This project demonstrates the implementation of Simple Linear Regression and Multiple Linear Regression using Python in Google Colab.
The goal is to understand how regression models predict dependent variables based on one or more independent variables.

🧠 Concepts Covered

Simple Linear Regression

Multiple Linear Regression

Data Preprocessing

Train-Test Split

Model Training

Model Evaluation (R² Score, MSE)

Data Visualization

🛠️ Technologies Used

Python

Google Colab

NumPy

Pandas

Matplotlib

Scikit-learn

📂 Project Structure
Linear-Regression-Project/
│
├── Linear_Regression.ipynb
├── Multiple_Regression.ipynb
└── README.md

📈 Simple Linear Regression

Simple Linear Regression predicts a dependent variable (Y) using one independent variable (X).

Example:

Predicting Salary based on Years of Experience.

Steps Performed:

Import libraries

Load dataset

Split dataset into training and testing sets

Train the Linear Regression model

Predict results

Visualize results

Evaluate model performance

📊 Multiple Linear Regression

Multiple Linear Regression predicts a dependent variable using multiple independent variables.

Example:

Predicting House Price based on:

Area

Number of Bedrooms

Location Score

Steps Performed:

Import libraries

Load dataset

Data preprocessing

Train-Test split

Train regression model

Predict values

Evaluate model using R² Score and Mean Squared Error

📉 Model Evaluation Metrics

R² Score – Measures how well the model explains the data.

Mean Squared Error (MSE) – Measures average squared difference between actual and predicted values.

▶️ How to Run the Project

Open Google Colab.

Upload the notebook file (.ipynb).

Run all cells.

Modify dataset if needed.

📌 Sample Code Snippet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
dataset = pd.read_csv("data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

🎯 Learning Outcomes

Understand regression concepts

Build prediction models

Evaluate model performance

Visualize regression results

