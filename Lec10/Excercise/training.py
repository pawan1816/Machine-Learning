import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("titanic.csv")

# print(df.head())





features = [ 'Sex','Age','Pclass','Fare']  # Add more numerical features if you have
target = 'Survived'

# Convert categorical features to numerical (e.g., 'Sex')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Handle missing values in 'Age' if any
df['Age'] = df['Age'].fillna(df['Age'].mean())

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(df[features],df[target],test_size=0.3, random_state=42)


print(X_train.head())
print(Y_train.head())


# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.barplot(x='Sex', y='Survived', data=df)
# plt.title('Survival Rate by Gender')
# plt.xlabel('Sex (0 = Male, 1 = Female)')
# plt.ylabel('Survival Rate')
# plt.grid(True)
# plt.show()


# sns.barplot(x='Pclass', y='Survived', data=df, palette='Set2')
# plt.title('Survival Rate by Passenger Class')
# plt.xlabel('Passenger Class')
# plt.ylabel('Survival Rate')
# plt.grid(True)
# plt.show()


# plt.hist(df[df['Survived'] == 0]['Age'], bins=20, alpha=0.5, label='Not Survived', color='red')
# plt.hist(df[df['Survived'] == 1]['Age'], bins=20, alpha=0.5, label='Survived', color='green')
# plt.legend()
# plt.title('Age Distribution: Survived vs Not Survived')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.grid(True)
# plt.show()


# sns.boxplot(x='Survived', y='Fare', data=df, palette='coolwarm')
# plt.title('Fare Distribution by Survival')
# plt.xlabel('Survived')
# plt.ylabel('Fare')
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette={0: 'red', 1: 'green'})
# plt.title('Fare vs Age (colored by Survival)')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(8,6))
# sns.heatmap(df[['Sex', 'Age', 'Pclass', 'Fare', 'Survived']].corr(), annot=True, cmap='viridis')
# plt.title('Correlation Heatmap')
# plt.show()




from sklearn import tree
model=tree.DecisionTreeClassifier()


model.fit(X_train,Y_train)


model.score(X_test,Y_test)

prdt=model.predict([[0,29.699118,3, 7.8958]])

print(prdt)

import joblib

joblib.dump(model,"implementation_joblib.pkl")






