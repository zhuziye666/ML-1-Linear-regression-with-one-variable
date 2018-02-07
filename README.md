# ML-1-Linear-regression-with-one-variable
the code with python will show in the following

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#导入数据并可视化
path='ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
print(data.head())
print(data.describe())
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
plt.show()
#代价函数costfunction
def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
data.insert(0,'Ones',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
print(X.head())
print(y.head())
X=np.matrix(X.values)#(97,2)
y=np.matrix(y.values)#(971)
theta=np.matrix(np.array([0,0]))#(1,2)
print(X.shape,y.shape,theta.shape)
print(computeCost(X,y,theta))
#梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    m=len(y)
    J_history=np.zeros(iters)
    for i in range(iters):
        inner1=(X*theta.T-y).T*X
        theta=theta-(alpha/m*(inner1))
        J_history[i]=computeCost(X,y,theta)
    return theta, J_history
alpha=0.01
iters=1500
theta,j=gradientDescent(X,y,theta,alpha,iters)
print(theta,j[iters-1])
predict1=[1,3.5]*theta.T
print('For population=35,000, we predict a profit of ',predict1*10000)
x=np.linspace(data.Population.min(),data.Population.max(),100)
f=theta[0,0]+theta[0,1]*x

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),j,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title("Error vs.Training Epoch")
plt.show()
        
