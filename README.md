# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. import the libraries

2. initialize the scaler of X and Y
 
3. train and test the data of X and Y
 
4. Print the Mean Square Error

## Program:

Program to implement the the Logistic Regression Model to Predict the fetch_california_housing

Developed by: Preethi S

RegisterNumber:  212223230157


```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
X = data.data[:, :3]
Y=np.column_stack((data.target, data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_Y=StandardScaler()
scaler_X=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

```
![image](https://github.com/user-attachments/assets/0548b0e0-ee14-4e80-a956-d3a25cd3024f)
```

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
```

![image](https://github.com/user-attachments/assets/e8915b66-16b7-49b8-9b41-e0ac568627e4)
```
print("\nPredictions:\n",Y_pred[:5])
```

![image](https://github.com/user-attachments/assets/a31bb097-6386-48ec-a22c-e0276bebd3eb)
```
```


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
