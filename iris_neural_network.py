import numpy as np
import scipy.special as spspl
import pandas as pd
import random
import csv

class NN:

    # constructor asks for number of inputs, number of hidden neurons, number of outputs
    def __init__(self, iSize, hSize, oSize):
        self.iSize = iSize;
        self.hSize = hSize;
        self.oSize = oSize;

        # initialize the weights between input and hidden layer in the form of matrix
        self.ihWeights = np.random.rand(hSize, iSize)
        
        # initialize the weights between hidden and output layer in the form of matrix
        self.hoWeights = np.random.rand(oSize, hSize)
        
        # initialize the bias for hidden layer
        self.hBias = np.random.rand(hSize)

        # initialize the bias for output layer
        self.oBias = np.random.rand(oSize)

    def feedForward(self,inputs):
        # calculate the sigmoid of (sum of bias and (matrix multiplication of weights and input))
        # calculate the matrix multiplication of input-hidden-Weights and inputs, then add hidden-bias,
        # calculate the sigmoid(activation function) to get the hidden neuron values
        hNeurons = spspl.expit(np.add(np.dot(self.ihWeights, inputs), self.hBias))

##        print()
##        print("Hidden Neurons")
##        print(hNeurons)
##        
        #calculate the matrix multiplication of hidden-output-Weights and hidden-neurons, then add output-bias,
        # calculate the sigmoid(activation function) to get the output values
        outputs = spspl.expit(np.add(np.dot(self.hoWeights, hNeurons), self.oBias))

##        print()
##        print("Output Neurons")
##        print(outputs)

        return outputs
        
    def train(self,inputs, targets):
        # calculate the sigmoid of (sum of bias and (matrix multiplication of weights and input))
        # calculate the matrix multiplication of input-hidden-Weights and inputs, then add hidden-bias,
        # calculate the sigmoid(activation function) to get the hidden neuron values
        hNeurons = spspl.expit(np.add(np.dot(self.ihWeights, inputs), self.hBias))

##        print()
##        print("Hidden Neurons")
##        print(hNeurons)

        
        #calculate the matrix multiplication of hidden-output-Weights and hidden-neurons, then add output-bias,
        # calculate the sigmoid(activation function) to get the output values
        outputs = spspl.expit(np.add(np.dot(self.hoWeights, hNeurons), self.oBias))
##        print()
##        print("Output Neurons")
##        print(outputs)

        
        # calculate the errors at output layer (Errors at output is difference of actual and calculated)
        oErrors = np.subtract(targets, outputs)
##        print()
##        print("Errors")
##        print(oErrors);
        
        
        # calculate the errors at hidden layer (We have weights between hidden & output and erros at output layer)
        # So we transpose the (hidden-output) weights, and multiply with the errors to calculate the errors at hidden layer
        hErrors = np.dot(self.hoWeights.T, oErrors)
##        print()
##        print("HErrors")
##        print(hErrors)

        # calculating the gradiant for outputs. Derivative of outputs
        ogradients = 0.1 * np.multiply(oErrors, np.multiply(outputs, (1 - outputs)))
##        print()
##        print("Gradients")
##        print(ogradients)
        
        dHOWeights = np.dot(np.reshape(ogradients, (len(ogradients),1)), np.reshape(hNeurons,(len(hNeurons),1)).T)

        
        # update hidden output weights
        self.hoWeights = np.add(self.hoWeights, dHOWeights)
##        print()
##        print("New Hidden-Output")
##        print(self.hoWeights)

        # update output bias 
        self.oBias = np.add(self.oBias, ogradients)
##        print()
##        print("New Bias Output")
##        print(self.oBias)

        #repeat same delta weight calculation for input hidden weights
        #dIHWeights = 0.1*error*input
        hgradients = 0.1 * np.multiply(hErrors, np.multiply(hNeurons, (1 - hNeurons)))
        
        dIHWeights = np.dot(np.reshape(hgradients, (len(hgradients),1)), np.reshape(inputs, (len(inputs),1)).T)

        # update input hidden weights
        self.ihWeights = np.add(self.ihWeights, dIHWeights)
##        print()
##        print("New Input-Hidden")
##        print(self.ihWeights)

        # update hidden bias
        self.hBias = np.add(self.hBias, hgradients)
##        print()
##        print("New Bias Hidden")
##        print(self.hBias)
        

nn = NN(4, 5, 3)

setosa = []
versicolor = []
virginica = []

with open('iris.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        fType = row[4]
        if fType == "setosa":
            setosa.append(row)
        elif fType == "versicolor":
            versicolor.append(row)
        elif fType == "virginica":
            virginica.append(row)


# shuffle data
random.shuffle(setosa)
random.shuffle(versicolor)
random.shuffle(virginica)


print(len(setosa))
print(len(versicolor))
print(len(virginica))

train = []
train.extend(setosa[:40])
train.extend(versicolor[:40])
train.extend(virginica[:40])
random.shuffle(train)

test = []
test.extend(setosa[40:])
test.extend(versicolor[40:])
test.extend(virginica[40:])

print(len(train))
#print(train)

print(len(test))
#print(test)

for i in range(49999):
    row = random.choice(train)
    inputs = np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
    outputs = []
    if row[4] == "versicolor":
        outputs = np.array([0,1,0])
    elif row[4] == "virginica":
        outputs = np.array([0,0,1])
    elif row[4] == "setosa":
        outputs = np.array([1,0,0])
    nn.train(inputs, outputs)

for row in test:
    inputs = np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
    outputs = []
    if row[4] == "versicolor":
        outputs = np.array([0,1,0])
    elif row[4] == "virginica":
        outputs = np.array([0,0,1])
    elif row[4] == "setosa":
        outputs = np.array([1,0,0])
    print(outputs)
    print(nn.feedForward(inputs))
    print()
