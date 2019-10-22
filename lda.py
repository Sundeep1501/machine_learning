import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

X1 = np.array([4,2,2,3,4,9,6,9,8,10])
X2 = np.array([1,4,3,6,4,10,8,5,7,8])
X3 = np.array(['a','a','a','a','a','b','b','b','b','b'])

df = DataFrame()
df['X1'] = X1
df['X2'] = X2
df['X3'] = X3

print (df)

## calculate the mean of each feature
ma = np.array([np.mean(df.loc[df['X3']=='a',['X1']]), np.mean(df.loc[df['X3']=='a',['X2']])])
mb = np.array([np.mean(df.loc[df['X3']=='b',['X1']]), np.mean(df.loc[df['X3']=='b',['X2']])])

## scatter matrix for each class
Sa = np.zeros((2,2))
Sb = np.zeros((2,2))
for idx, val in enumerate(X1):
    p = np.array([[X1[idx]],[X2[idx]]])
    if(X3[idx]=='a'):
        Sa += np.dot(p-ma,(p-ma).T)
    else:
        Sb += np.dot(p-mb,(p-mb).T)

## scatter matrix between class
Sw = Sa + Sb

## projection vector
SwI = np.linalg.inv(Sw)
W = np.dot(SwI,ma-mb)
W = W.T
print (W)

## new feature value
Y1 = np.zeros(X1.size)
for idx, val in enumerate(X1):
    p = np.array([[X1[idx]],[X2[idx]]])
    Y1[idx] = np.dot(W,p) * -1
    print(Y1[idx])
df['Y1'] = Y1

plt.figure(figsize=(7,7))
plt.scatter(df.loc[df['X3']=='a',['X1']], df.loc[df['X3']=='a',['X2']], color='r', marker='*', label='a')
plt.scatter(df.loc[df['X3']=='b',['X1']], df.loc[df['X3']=='b',['X2']], color='g', marker='o', label='b')

plt.scatter(df.loc[df['X3']=='a',['Y1']], np.zeros(5), color='r', marker='*')
plt.scatter(df.loc[df['X3']=='b',['Y1']], np.zeros(5), color='g', marker='o')

plt.legend(loc='best')
plt.show()


