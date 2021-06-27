# -*- coding: utf-8 -*-

#import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#read MNIST dataset and create multiple datasets
data_train=pd.read_csv("sample_data//mnist_train_small.csv",  header = None)
print(data_train)
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))

def data_copy(data):
    D0=data.copy()
    D1= data.copy() 
    D2= data.copy() 
    D3= data.copy() 
    D4= data.copy() 
    D5= data.copy() 
    D6= data.copy() 
    D7=data.copy() 
    D8= data.copy() 
    D9= data.copy()
    datasets=[D0,D1,D2,D3,D4,D5,D6,D7,D8,D9]
    return datasets

datasets=data_copy(data_train)
print("Datasets:"datasets)

#create individual binary datasets
def individual_binary_datasets(data,datasets):
    for i in range(10):
        for j in range(len(data.T.columns)):
            if datasets[i].at[j,0]==i:
                datasets[i].at[j,0]=1
            else:
                datasets[i].at[j,0]=0
    return datasets

datasets=individual_binary_datasets(data_train,datasets)

#dictionary holding dataset names as keys and their input and output splits as value pairs
def data_dic_creator(datasets):    
    TrainTest_datadict={}
    for i in range(10):
        x=datasets[i].iloc[:,1:]
        x=minmax.fit_transform(x)
        x=x.T
        y=datasets[i].iloc[:,0]
        y=np.array([y])
        print(x)
        TrainTest_datadict['D'+str(i)]=[x,y]
    return TrainTest_datadict

TrainTest_datadict=data_dic_creator(datasets)

#logistic function
    def hypothesis_cal(weights,X,bias):
        z = np.dot(weights,X) + bias 
        hypothesis = np.exp(z)/(1 + np.exp(z))
        return hypothesis

#cost function
def cost_function(hypothesis,m,y,k,costfunc_values):
    j = 1/m*(-1*(np.sum(y*np.log(hypothesis) + (1-y)*np.log(1-hypothesis))))
    costfunc_values.append(j)
    k+=1
    return j,k,costfunc_values,costfunc_values

#gradient descent
def gradient_descent(hypothesis,y,X,weights,bias,alpha,m):
    dw =  1/m * np.dot(hypothesis-y,X.T)
    db =  1/m * np.sum(hypothesis-y)
    weights = weights - alpha*dw
    bias = bias - alpha*db
    return weights,bias

print(TrainTest_datadict['D'+str(0)][0].T[0:100].shape)

def logistic_regression(TrainTest_datadict,data):    
    iterations = 1000000   
    alpha = 0.05
    m = len(data.T.columns)
    cost_values = []
    trained_parameters = []
    for trainsets in range(10):
        X = TrainTest_datadict['D'+str(trainsets)][0] 
        Y = TrainTest_datadict['D'+str(trainsets)][1]
        weights = np.random.randn(1,len(data.columns)-1)
        bias = 0 
        costfunc_values = []
        k = 0
        print('Training for dataset '+str(trainsets+1))
        for i in range(1,iterations+1):
          for j in range(1,len(X)//100+1):
            x=X.T[(j-1)*100:j*100]
            y=Y.T[(j-1)*100:j*100]
            x=x.T
            y=y.T
            hypothesis=hypothesis_cal(weights,x,bias)
            weights,bias=gradient_descent(hypothesis,y,x,weights,bias,alpha,m)

          hypothesis=hypothesis_cal(weights,X,bias)
          j,k,costfunc_values,costfunc_values=cost_function(hypothesis,m,Y,k,costfunc_values)
          if i%2000 == 0:
              print('running @ ',j)
          if i%2 == 0:
              if abs(j-costfunc_values[-2])<0.000001:
                  if abs(j-costfunc_values[-3])<0.000001:
                      break 
        cost_values.append(costfunc_values)  
        trained_parameters.append([weights,bias])      
        print('iteration number:',k)
    return trained_parameters,cost_values

trained_parameters,cost_values=logistic_regression(TrainTest_datadict,data_train)

#prediction of accuracy for each dataset
def accuracy_for_each_dataset(TrainTest_datadict,trained_parameters):
    for datasetnum in range(10):
        x = TrainTest_datadict['D'+str(datasetnum)][0]
        y = TrainTest_datadict['D'+str(datasetnum)][1]
        x=x.T
        x=pd.DataFrame(x)
        weights = trained_parameters[datasetnum][0]
        bias = trained_parameters[datasetnum][1]
        correct_predictions = 0 
        for i in range(len(data_train.T.columns)):
            z = np.dot(weights,x.iloc[i,:])+bias
            hypothesis = 1/(1 + np.exp(-z))
            if np.logical_and(hypothesis >= 0.5,y.T[i,0] == 1):
                correct_predictions+=1
            if np.logical_and(hypothesis < 0.5,y.T[i,0] == 0):
                correct_predictions+=1    
        #print(correct_predictions)      
        acc = (correct_predictions/len(data_train.T.columns))*100
        print('accuracy for dataset '+str(datasetnum+1)," = ",acc)

accuracy_for_each_dataset(TrainTest_datadict,trained_parameters)

def accurate_predictions(data,trained_parameters):
    inputs = data.iloc[:,1:]
    inputs = inputs.T
    outputs = data.iloc[:,0] 
    outputs = np.array(outputs)
    accuratepredicts = 0                
    for i in range(len(data.T.columns)):
        probabilities = []
        for j in range(10):
            weights = trained_parameters[j][0]
            bias = trained_parameters[j][1]
            z = np.dot(weights,inputs.iloc[:,i].T)+bias
            hypothesis = np.exp(z)/(1 + np.exp(z)) 
            probabilities.append(hypothesis)
        predict = probabilities.index(max(probabilities)) 
        if outputs.T[i] == predict:
            accuratepredicts+=1
    print(accuratepredicts)

    return accuratepredicts

def accuracy(accuratepredicts,data):
    print(accuratepredicts/len(data.T.columns)*100)


#cost function plots 
def plot_graph(cost_values):
    fig,a =  plt.subplots(2,5,figsize=(35,15))
    a[0][0].plot(cost_values[0])
    a[0][0].set_title("0's dataset")
    a[0][1].plot(cost_values[1])
    a[0][1].set_title("1's dataset")
    a[0][2].plot(cost_values[2])
    a[0][2].set_title("2's dataset")
    a[0][3].plot(cost_values[3])
    a[0][3].set_title("3's dataset")
    a[0][4].plot(cost_values[4])
    a[0][4].set_title("4's dataset")
    a[1][0].plot(cost_values[5])
    a[1][0].set_title("5's dataset")
    a[1][1].plot(cost_values[6])
    a[1][1].set_title("6's dataset")
    a[1][2].plot(cost_values[7])
    a[1][2].set_title("7's dataset")
    a[1][3].plot(cost_values[8])
    a[1][3].set_title("8's dataset")
    a[1][4].plot(cost_values[9])
    a[1][4].set_title("9's dataset")
    plt.show()

plot_graph(cost_values)

#test dataset
data_test=pd.read_csv("C:\\Users\\ayush\\mnist_test.csv",header=None)
accuratepredicts=accurate_predictions(data_test,trained_parameters)
accuratepredicts
accuracy(accuratepredicts,data_test)

