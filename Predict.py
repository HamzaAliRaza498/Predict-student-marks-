import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data for study hours and corresponding grades
Data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Grade': [33, 40, 45, 50, 60, 65, 70, 75, 85, 90]
}

# Creating a DataFrame
df = pd.DataFrame(Data)

# Splitting the data into feature (X) and target (y)
x = df[['Hours']]  # Feature (Hours), it's 2D
y = df[['Grade']]  # Target (Grade), it's also 2D

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating a Linear Regression model
Model = LinearRegression()
Model.fit(x_train, y_train)

# Taking user input for the number of study hours
user_input = float(input("Enter a Studying hour: "))

# Predicting the grade based on input hours
Predicted = Model.predict([[user_input]])

# Printing the predicted grade
print(f"Predicted Exam Score is: {Predicted[0][0]:.2f}")
