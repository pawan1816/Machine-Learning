import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("carprices.csv")

# Strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print to verify
print(df.head())

# Define features and target
features = ['Mileage', 'Age(yrs)']  # Add more numerical features if you have
target = 'Sell Price($)'

# # Plot
# plt.figure(figsize=(12, 5))

# for i, feature in enumerate(features):
#     plt.subplot(1, len(features), i + 1)  # subplot: 1 row, len(features) columns
#     plt.scatter(df[feature], df[target], color='green', edgecolors='black')
#     plt.xlabel(feature)
#     plt.ylabel('Sell Price($)')
#     plt.title(f'{feature} vs Price')
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

from sklearn.model_selection import train_test_split

X = df[['Mileage','Age(yrs)']]         # Features should be a 2D array (use double brackets)
y = df['Sell Price($)']     # Target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))



from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train)


print(X_test)
prdt=clf.predict(X_test)
print(prdt)
print(Y_test)
acry=clf.score(X_test,Y_test)

print(acry)