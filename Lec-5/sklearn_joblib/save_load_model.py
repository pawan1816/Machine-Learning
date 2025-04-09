import pandas as pd
import numpy as np

from sklearn import linear_model
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode


#load dataframe
df= pd.read_csv("Area_vs_Price.csv")
# print(df)


# plt.figure(figsize=(8, 6))  # Set figure size
# plt.scatter(df["area"], df["price"], color="blue", alpha=0.5)

# # Add labels and title
# plt.xlabel("X-Axis Label")
# plt.ylabel("Y-Axis Label")
# plt.title("Scatter Plot of area vs price")

# # Show the plot
# plt.show(block=True)


# create a new dataframe
# it is for the linear model class , linear model object fit method() expect the 2D dataframe. for creating Object and calling the fit method()
new_df=df.drop('price', axis='columns')
print(new_df)


#Creating a LinearRegression object & traning the model
model = linear_model.LinearRegression()
model.fit(new_df,df.price)


# prediction

predicted_price = model.predict(pd.DataFrame([[3300]], columns=["area"]))
print("Predicted Price for 3300 sq ft:", predicted_price[0])





# to save the file
import joblib
import pandas as pd

# Save the trained model
joblib.dump(model, 'model_joblib.pkl')



