# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Syed Mohamed Raihan M
RegisterNumber:  212224240167

import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()


print("data.info()")
df.info()


print("data.isnull().sum()")
df.isnull().sum()


print("data value counts")
df["left"].value_counts()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()


print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()


y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()

```

## Output:

![Screenshot 2025-05-19 094810](https://github.com/user-attachments/assets/bec313a1-ee57-4024-b93c-15638e18ca83)

![Screenshot 2025-05-19 094824](https://github.com/user-attachments/assets/fd2045eb-a5cc-4b24-a209-8afc4b7efb37)

![Screenshot 2025-05-19 094834](https://github.com/user-attachments/assets/5f77dc57-29b7-4f1b-a498-aa6518e18350)
![Screenshot 2025-05-19 094841](https://github.com/user-attachments/assets/bf61aa9f-357b-45d3-ba74-fe058cf55ff5)
![Screenshot 2025-05-19 094851](https://github.com/user-attachments/assets/f81ed5ef-6790-411a-9c61-c5854d248fd1)
![Screenshot 2025-05-19 094901](https://github.com/user-attachments/assets/5dd91932-f41b-4115-8f6c-918dbfbdcf8e)
![Screenshot 2025-05-19 094913](https://github.com/user-attachments/assets/a3b2c763-0ab6-40eb-84f3-8abd8250d328)
![Screenshot 2025-05-19 094921](https://github.com/user-attachments/assets/22a1bd27-bcc9-49f6-8ec7-b2aab3ecc734)
![Screenshot 2025-05-19 094934](https://github.com/user-attachments/assets/3a801b40-f1fa-4b90-b4e0-67b81cae28be)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
