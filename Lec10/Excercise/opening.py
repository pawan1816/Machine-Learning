import joblib

mp=joblib.load("implementation_joblib.pkl")


predicted_price = mp.predict([[0,43,2,26.2500]])
print("Predicted Servival:", predicted_price[0])