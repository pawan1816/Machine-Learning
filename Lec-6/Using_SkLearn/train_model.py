import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel("data.xlsx", engine='openpyxl')

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Encode the town column using LabelEncoder
le = LabelEncoder()
dfle = df.copy()
dfle['town'] = le.fit_transform(dfle['town'])

# Features and target
X = dfle[['town', 'area']].values
Y = dfle['price'].values

# Apply OneHotEncoder on the first column (town)
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [0])
], remainder='passthrough')  # Keep the 'area' column as is

X = ct.fit_transform(X)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Making predictions
# To make predictions, you need to know the one-hot encoded position of each town.
# You can check the encoding via:
print("Encoded town categories:", le.classes_)

# Example: predicting price for a specific encoded input
# Suppose towns = ['Mumbai', 'Pune', 'Delhi'], and le.classes_ gives ['Delhi', 'Mumbai', 'Pune']
# Then encoding for Delhi = [1, 0, 0], Mumbai = [0, 1, 0], Pune = [0, 0, 1]

# Predict price for 'Pune' with area 3400 (Pune = [0, 0, 1])
print(model.predict([[0, 0, 1, 3400]]))

# Predict price for 'Mumbai' with area 2000 (Mumbai = [0, 1, 0])
print(model.predict([[0, 1, 0, 2000]]))


accurecy=model.score(X,Y)

print(accurecy)
      

