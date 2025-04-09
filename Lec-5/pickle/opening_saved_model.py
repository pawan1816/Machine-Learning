import pickle
import pandas as pd

# Load the saved model
with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

# Create input data with correct column name
data = pd.DataFrame([[3300]], columns=["area"])

# Predict
pred = mp.predict(data)
print(pred)
