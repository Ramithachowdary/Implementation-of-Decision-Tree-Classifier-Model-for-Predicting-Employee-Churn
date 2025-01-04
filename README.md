# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score 


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ramitha chowdary s
RegisterNumber:  24900704

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree  # Import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

*/
```

## Output:
![Screenshot 2024-11-28 201426](https://github.com/user-attachments/assets/657a253a-8fd3-4993-8fb3-7c63739c0e86)
![Screenshot 2024-11-28 201514](https://github.com/user-attachments/assets/f1350406-b3dd-401e-a2c8-56e846f52a2e)
![Screenshot 2024-11-28 201543](https://github.com/user-attachments/assets/314acf02-2691-4161-bf52-a211c3b24dc0)
```
0    11428
1     3571
Name: left, dtype: int64
```
![Screenshot 2024-11-28 201718](https://github.com/user-attachments/assets/d03adb5f-519a-4051-b471-1f4354dfa06c)
![Screenshot 2024-11-28 201801](https://github.com/user-attachments/assets/987387f1-6c29-4b7c-807c-255838591714)

0.9846666666666667
![Screenshot 2024-11-28 201834](https://github.com/user-attachments/assets/75bc9118-c203-4c43-a3b6-9604314147e0)

![Screenshot 2024-11-28 142058](https://github.com/user-attachments/assets/686825b1-2b4a-4c87-963a-24f0887ca467)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
