# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:V.KAILASH
RegisterNumber:24001383



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:


Dataset

![Screenshot (107)](https://github.com/user-attachments/assets/5f03b0a0-7507-435e-8f36-3aa93ba097f1)



Head Values

![Screenshot (108)](https://github.com/user-attachments/assets/0513590b-559e-43a4-ab52-a9f1b06adc6b)


Tail Values

![Screenshot (109)](https://github.com/user-attachments/assets/fb44c489-8012-4261-a945-e41dab5d2733)


X and Y Values


![Screenshot (110)](https://github.com/user-attachments/assets/e9bd2a43-9f04-4db5-9484-d333c5da2538)



Prediction values of X and Y


![Screenshot (111)](https://github.com/user-attachments/assets/64ff71a9-4e81-436b-827d-ce23fb6c5898)


MSE,MAE and RMSE


![Screenshot (115)](https://github.com/user-attachments/assets/63530946-c593-4421-86d8-af9b0523974a)



Training Set


![Screenshot (112)](https://github.com/user-attachments/assets/dd36b3e3-a41f-41a9-8f47-06aaacb6b930)


Testing Set


![Screenshot (114)](https://github.com/user-attachments/assets/e60db5ba-9d32-464e-8e2d-0eb827eaa835)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
