
import joblib
import pandas as pd

# ✅ Load the model
mp = joblib.load('model_joblib.pkl')

# Input data
from sklearn import datasets

# Load the digits dataset
digits = datasets.load_digits()
# Predict
predicted_price = mp.predict([digits.images[67].reshape(64)])
print("Predicted price:", predicted_price[0])

print(digits.target[67])