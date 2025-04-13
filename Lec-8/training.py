import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("insurance_data.csv")

print(df)

plt.figure(figsize=(12, 5))

plt.scatter(df["age"], df["bought_insurance"], color='green', edgecolors='black')
plt.xlabel("age")
plt.ylabel('bought_insurance')
plt.title(f'{"age"} vs bought_insurance')
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df[["age"]], df["bought_insurance"], test_size=0.3, random_state=42)


print(X_test)
print(len(X_test))

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,Y_train)


print(model.predict(X_test))


# probablity predication

model.predict_proba(X_test)


model.score(X_test,Y_test)

model.predict_proba(X_test,Y_test)

# to save the file
import joblib
import pandas as pd

# Save the trained model
joblib.dump(model, 'model_joblib.pkl')