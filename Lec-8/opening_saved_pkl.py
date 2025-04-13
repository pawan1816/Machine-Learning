
import joblib
import pandas as pd

# ✅ Load the model
mp = joblib.load('model_joblib.pkl')

# Input data
data = pd.DataFrame([[33]], columns=["age"])

# Predict
predicted_price = mp.predict(data)
print("Predicted price:", predicted_price[0])