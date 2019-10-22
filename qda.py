import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


X = np.array([[4,4],[6,6],[4,6],[6,4],[5,5],[2,8],[4,8],[6,8],[8,8],[8,6],[8,4],[8,2]])
y = np.array([1,1,1,1,1,2,2,2,2,2,2,2])
clf = QuadraticDiscriminantAnalysis()
#clf.set_params("{store_covariance: True}");
m = clf.fit(X,y)
print(m.get_params())
print(m.priors_)
print(m.means_)
#print(m.covariance_)

x = np.array([[5,6]])

print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(m.predict_log_proba(x)))
x = np.array([[2,6]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(m.predict_log_proba(x)))
x = np.array([[3,3]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[4,7]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[7,5]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[6.5,5]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[6.6,5]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[6.7,5]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[6.8,5]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
x = np.array([[6.9,5]])
print(str(x)+" is class "+str(clf.predict(x)) +" per class "+str(clf.predict_log_proba(x)))
##############################


P1 = np.array([4,6,4,6,5,2,4,6,8,8,8,8])
P2 = np.array([4,6,6,4,5,8,8,8,8,6,4,2])
P3 = np.array(['h','h','h','h','h','c','c','c','c','c','c','c'])

pdf = DataFrame()
pdf['P1'] = P1
pdf['P2'] = P2
pdf['P3'] = P3

print (pdf)

plt.figure(figsize=(7,7))
plt.scatter(pdf.loc[pdf['P3']=='h',['P1']], pdf.loc[pdf['P3']=='h',['P2']], color='r', marker='h', label='h')
plt.scatter(pdf.loc[pdf['P3']=='c',['P1']], pdf.loc[pdf['P3']=='c',['P2']], color='b', marker='h', label='c')


plt.legend(loc='best')
plt.show()


