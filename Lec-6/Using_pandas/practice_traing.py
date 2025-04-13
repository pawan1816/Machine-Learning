import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data (or replace with your Excel data)
df = pd.read_csv("Practice_data.csv")

# Scatter plot: area vs. price
# Features to compare vs price
# Optional: Normalize column names
df.columns = df.columns.str.strip()

# One-hot encode the first column ('Car Model')
df_encoded = pd.get_dummies(df, columns=['Car Model'])

# Show the updated DataFrame
df_dummies=pd.concat([df,df_encoded], axis='columns')

print(df_dummies)
features = ['Mileage', 'Age(Yrs)']  # Add more numerical features if you have
target = 'Sell Price($)'

# Plot
plt.figure(figsize=(12, 5))

for i, feature in enumerate(features):
    plt.subplot(1, len(features), i + 1)  # subplot: 1 row, len(features) columns
    plt.scatter(df[feature], df[target], color='green', edgecolors='black')
    plt.xlabel(feature)
    plt.ylabel('Sell Price($)')
    plt.title(f'{feature} vs Price')
    plt.grid(True)

plt.tight_layout()
plt.show()



# # below code does not working for this model becouse the difference between the numbers of Milage and Age(Yrs) is very high
# # # Extract target and features
# # target = 'Sell Price($)'
# # features = ['Mileage', 'Age(Yrs)']  # Exclude 'Car Model' (categorical)

# # # Plot all features vs. price in one graph
# # plt.figure(figsize=(10, 6))

# # for feature in features:
# #     plt.scatter(df[feature], df[target], label=feature)

# # plt.xlabel("Feature Values")
# # plt.ylabel("Sell Price ($)")
# # plt.title("Features vs Sell Price (All in One Plot)")
# # plt.legend()
# # plt.grid(True)
# # plt.show()