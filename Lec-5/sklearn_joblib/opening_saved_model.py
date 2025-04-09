
import joblib
import pandas as pd

# ✅ Load the model
mp = joblib.load('model_joblib.pkl')

# Input data
data = pd.DataFrame([[3300]], columns=["area"])

# Predict
predicted_price = mp.predict(data)
print("Predicted price:", predicted_price[0])