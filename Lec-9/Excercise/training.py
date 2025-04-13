from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(dir(iris))

import seaborn as sns
import pandas as pd

# Convert to DataFrame for visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Map target to species name
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Pairplot
sns.pairplot(df, hue="species")
plt.show()
