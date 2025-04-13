
import pandas as pd
df= pd.read_csv("salaries.csv")
print(df.head())

inputs= df.drop('salary_more_then_100k',axis='columns')
target= df['salary_more_then_100k']

print(inputs)
print(target)


from sklearn.preprocessing import LabelEncoder

le_company=LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])

inputs_n=inputs.drop(['company','job','degree'],axis='columns')
print(inputs)
print(inputs_n)


from sklearn import tree
model =tree.DecisionTreeClassifier()

model.fit(inputs_n,target)


prdt=model.score(inputs_n,target)

print(prdt)

iPrdt=model.predict([[2,0,1]])

print(iPrdt)


import joblib
import pandas as pd

# Save the trained model
joblib.dump(model, 'model_joblib.pkl')