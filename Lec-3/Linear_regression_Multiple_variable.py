import pandas as pd
import numpy as np


from sklearn import linear_model
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode


#load dataframe
df= pd.read_csv("data.csv")

print(df)


import math

# print(df.info())
# print(df.head())
# print("bedrooms column:", df['bedrooms'])
# print("median:", df['bedrooms'].median())


df.bedrooms=df.bedrooms.fillna(df['bedrooms'].median())
print(df)


#create model

model = linear_model.LinearRegression()

# train model

model.fit(df.drop('price', axis='columns'), df['price'])


# pridict
prdt=model.predict([[3000,3,40]])
print(prdt)

# coeficient

co=model.coef_
print(co)

# intercept

intcp=model.intercept_

print(intcp)

# the result was came through m1*co1+ m2*co2 + m3* co3+ intercept
