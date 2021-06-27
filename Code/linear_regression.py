# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df=pd.read_csv("C:\\Users\\ayush\\mnist_train_small.csv",  header = None, prefix="var")
test_df=pd.read_csv("C:\\Users\\ayush\\mnist_test.csv",  header = None, prefix="var")

x_train=train_df.drop(['var0'], axis=1).values
x_bias_train=np.ones((np.shape(x1)[0], 1))
print(x_train)

y1=train_df['var0']
y_train=y1.to_numpy()
y_train=np.vstack(y_train)
print((y_train))

print(train_df.shape)

print(x_train.shape)
print(y_train.shape)

def model(x_train,x_bias_train, y_train, alpha, iterations):
    m=y_train.size
    theta=np.zeros((784,1))
    cost_list=[] 
    theta_bias=0
    for i in range(0,iterations):
        y_pred=np.dot(x_train, theta)+x_bias_train*theta_bias
        
        cost=(1/(2*m))*np.sum((y_pred-y_train)**2)
        d_theta_bias=(1/m)*np.dot(x_bias_train.T,y_pred-y_train)
        d_theta=(1/m)*np.dot(x_train.T, y_pred-y_train)
        
        theta=theta-alpha*d_theta
        theta_bias=theta_bias-alpha*d_theta_bias    
        cost_list.append(cost)
        
    return theta, cost_list,theta_bias

iterations=10000
alpha=0.0000005
theta, cost_list,theta_bias=model(x_train,x_bias_train, y_train, alpha=alpha, iterations=iterations)

x_test=test_df.drop(['var0'], axis=1).values
x_test_bias=np.ones((np.shape(x_test)[0], 1))
print(x_test)

y2=test_df['var0']
y_test=y2.to_numpy()
y_test=np.vstack(y_test)
print(y_test)

print(cost_list)

y_pred=np.dot(x_test, theta)+x_test_bias*theta_bias
print(y_pred)

g=np.sum(np.square(y_test-y_pred))
y=np.sum(np.square(y_test-np.mean(y_test)))

print("r2 calculated is:", (1-(g/y)))

iterations_list=[int(x) for x in range(1,iterations+1,1)]
plt.scatter(cost_list, iterations_list)
