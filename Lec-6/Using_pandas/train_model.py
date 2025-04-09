import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel("data.xlsx", engine='openpyxl')

# Show the first few rows
# print(df.head())


dummies=pd.get_dummies(df.town)
# print(dummies)


# join dummies and df

df_dummies=pd.concat([df,dummies], axis='columns')

# print(df_dummies)


# drop town and multicollinear for model training because we can derive one of the  One HOT encoding

df_dummies.drop(['town','west_windsor'], axis='columns', inplace=True)

# print(df_dummies)

X= df_dummies.drop('price',axis='columns')
# print(X)

Y=df_dummies.price
# print(Y)

from sklearn.linear_model import LinearRegression
model =LinearRegression()
model.fit(X,Y)


prdt=model.predict([[3400,0,0]])
print(prdt)