import numpy as np
from matplotlib import pyplot as plt

def printMatrix(matrix):
    for row in matrix:
        print(row)
        
def calculateManhattanDistance(x1, y1, x2, y2):
    return abs(x1-x2) + abs(y1-y2)

def center(x1, y1, x2, y2):
    return [(x1+x2)/2,(y1+y2)/2]
    
X = [[5,3],  
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],]
K = 2

printMatrix(X)

while len(X) > K:
    distances = [[float("inf") for i in range(len(X))] for j in range(len(X))] 
    minIndex = [0,0]
    minDistance = float("inf")
    for i in range(len(X)):
        rowDis = []
        for j in range(len(X)):
            if j < i:
                d = calculateManhattanDistance(X[i][0],X[i][1],X[j][0],X[j][1])
                distances[i][j] = d
                if minDistance > d:
                    minDistance = d;
                    minIndex = [i,j]
            
    #printMatrix(distances)
    print()

    print('Merge')
    #print(X[minIndex[0]], X[minIndex[1]])
    print(minIndex)
    print()
    X[minIndex[0]] = center(X[minIndex[0]][0],X[minIndex[0]][1],X[minIndex[1]][0],X[minIndex[1]][1])
    del X[minIndex[1]]
    printMatrix(X)

