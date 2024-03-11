# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.ARJUN
RegisterNumber:  212222040012
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/b88ae6a0-e143-4db6-8a2f-f9423fdbc91b)

## Salary Data
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/914080d1-4429-4444-a9ea-96f121ffb48f)

## Data-status
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/5b40cc5b-6a9e-4bfe-959f-23cb28b982c6)

## Data duplicate
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/abdbc4ce-a3dd-4f2c-9743-3e41e1ce42f2)

## Print Data
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/0f50ea95-68c0-4742-90a0-0106ed7ed91e)

## Checking the null function()
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/ed700e30-b52a-49a5-90af-98b9930b0c3e)

## Y_Prediction Array
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/599ee5e1-e897-4b75-9a1a-72120e98ed7a)

## Accuracy value
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/f4058e15-f6a1-43a4-b9b5-56b267f48b3b)

## Confusion Value
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/16215026-e8d1-42b3-99ed-1d9b65c1cdb8)

## Classification Report
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/4dc0e37c-fcc1-487b-8512-51b245f99320)

## Prediction of LR
![image](https://github.com/ARJUN19122004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119429483/bfd5b3e4-24f1-4b4a-a065-ac3c705cfe8d)


















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
