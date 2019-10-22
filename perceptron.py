import numpy as np
import matplotlib.pyplot as plt

def activator(i,bias):
    if i>=(0+bias):
        return 1
    else:
        return -1

lrate = 0.2 # should be 0 to 1.

# input with 2 features, 2 Dimensional
X = np.array([[2,3],[2,5],[3,6],[1,3],[3,5],[3,1],[2,1],[3,3],[4,3],[4,4]])
#X = np.array([[1,3],[2,3],[3,3],[4,3],[5,3],[1,1],[2,1],[3,1],[4,1],[5,1]])
# Output feature to train
Y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
#Y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])

# generate random weights. one for each feature
# we are dealing with single layer, simple perceptron
W = np.random.uniform(size=len(X[0]))
print("Weights:" + str(W))

Bias = 0
#Bias = 2

while True:
    option = input("1:Training, 2:Predict, 3:Quit")
    if option == "1":
        for i in range(len(X)):
            sum = 0;
            for j in range(len(X[i])):
                sum = sum + X[i][j]*W[j]
            
            #activation function
            guess = activator(sum,Bias);
            actual = Y[i]
            error = actual - guess;
            print("Guess:%s, Actual:%s, Error:%s" %(str(guess), str(Y[i]), str(error)))

            # check error is zero or not
            if error != 0:
                #update weights
                for k in range(len(W)):
                    W[k] = W[k] + error*X[i][k] * lrate;
                print("Updated weights %s" %str(W));
    elif option == "2":
        sum = 0
        for i in range(len(W)):
            feature = float(input("input feature %d" %i));
            sum += feature* W[i]
        print("Belongs to %d" %activator(sum,Bias))
    else:
        break;
