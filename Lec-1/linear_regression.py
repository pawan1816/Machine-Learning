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



# x= model.intercept_
# y= model.coef_



#  for predicting the Unknown price of the data. test data are saved in test.csv and i am going to predict and store the result over there.

test_df=pd.read_csv("test.csv")
test_df.columns = ["area"]
test_df

# Predict the prices for the test dataset
test_df["predicted_price"] = model.predict(test_df)

# Save the updated test_df with predictions
test_df.to_csv("test.csv", index=False)

print("Predictions saved successfully to test.csv!")

