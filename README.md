# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the necessary python libraries
3.Read the dataset of Social_Network_Ads
4.Assign X and Y values for the dataset
5.Import StandardScaler for preprocessing of data
6.From sklearn library select the model to perform Logistic Regression
7.Using Gradient descent perform Logistic Regression on the dataset
8.Print the confusion matrix and accuracy
9.Display the data in graph using matplotlib libraries
10.Stop the program

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: B S SAI HARSHITHA
RegisterNumber: 212220040139

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Beauty Kumari S
RegisterNumber: 212220040023
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/content/Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
Output:
Sc_X
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)
#Fitting the Logistic Regression into the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)
#Predicting the test set results
Y_pred = classifier.predict(X_test)
Y_pred
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test,Y_pred)
print(confusion)
from sklearn import metrics
accuracy = metrics.accuracy_score(Y_test,Y_pred)
print(accuracy)
r_sen = metrics.recall_score(Y_test,Y_pred,pos_label = 1)
r_spec= metrics.recall_score(Y_test,Y_pred,pos_label = 0)
r_sen, r_spec
#Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.a
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape)
plt.xlim(X1.min(),X2.max())
plt.ylim(X1.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],c=ListedColormap(('red','green'))
(i),label=j)
plt.title("Logistic Regression(Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

*/
```

## Output:
![image](https://github.com/saiharshithabs/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/blob/133058441746692bd699775550a8e2be370479ef/WhatsApp%20Image%202022-10-14%20at%209.18.20%20AM.jpeg)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

