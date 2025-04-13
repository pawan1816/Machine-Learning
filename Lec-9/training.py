# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt

from sklearn import datasets

# Load the digits dataset
digits = datasets.load_digits()

print(dir(digits))
plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])

print(digits.target[0:5])

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.2,random_state=42)
print(len(X_train))
print(len(X_test))

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,Y_train)


acr=model.score(X_test,Y_test)
print(acr)



plt.matshow(digits.images[67])
print(digits.target[67])

print(model.predict([digits.data[67]]))


# digits.images[67] is a 2D image (8x8), but model.predict() expects a 1D array of 64 features (i.e., flattened image).

# To fix it, you need to reshape it properly.
print(model.predict([digits.images[67].reshape(64)]))


# Confusion matrix

y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(Y_test,y_predicted)
print(cm)



import seaborn as sn
# Plot
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

import joblib
import pandas as pd

# Save the trained model
joblib.dump(model, 'model_joblib.pkl')


