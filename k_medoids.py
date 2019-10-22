import numpy as np
import random
import math

X = np.array([2, 3, 3, 4, 6, 6, 7, 7, 8, 7])
Y = np.array([6, 4, 8, 7, 2, 4, 3, 4, 5, 6])
K = 2
medoids = []
allMedoids = []

oldMedoid = -1
oldCosts = []
oldDistances = []

# randomly choose K medoids(centers)
for i in range(K):
    medoids.append(random.randint(0, 9))
    allMedoids.append(medoids[i])

print(X)
print(Y)
print()
print(medoids)


while 1:
        
    distances = []
    costs = [0.0] * K
    
    for k in range(len(medoids)):
        # take each medoid(center)
        m = [X[medoids[k]], Y[medoids[k]]]
        d = []
        # calculate distance to every other point
        for i in range(X.size):
            p = [X[i], Y[i]]
            d.append(abs(m[0] - p[0]) + abs(m[1] - p[1]))
        distances.append(d)

    print(distances)


    # calculate the cost of K clusters
    for i in range(X.size):
        minIndex = -1 
        for k in range(len(medoids)):
            if(minIndex == -1):
                minIndex = k
            elif(distances[k][i] < distances[minIndex][i]):
                minIndex = k
        costs[minIndex] = costs[minIndex]+distances[minIndex][i]
    print(costs)

    # if already swapped, verify swap good.
    if(oldMedoid != -1):
        if(sum(costs) >= sum(oldCosts)):
            # bad swap, undo
            medoids[oldCosts.index(max(oldCosts))] = oldMedoid
            print('Bad swap undoing')
        else:
            # good swap
            print('Good swap undoing')
            oldCosts = costs
            oldDistances = distances
    else:
        print('Base Medoids')
        oldCosts = costs
        oldDistances = distances

    print(medoids)
    print(oldCosts)

    # choose next random non-medoid and swap with high cost medoid
    isCompleted = 1
    for i in range(X.size):
        newMedoid = i
        if newMedoid not in allMedoids:
            allMedoids.append(newMedoid)
            newMedoidIndex = oldCosts.index(max(oldCosts))
            oldMedoid = medoids[newMedoidIndex]
            medoids[newMedoidIndex] = newMedoid
            isCompleted = 0
            break
    
    print()
    print()
    print(medoids)
    if isCompleted:
        break
print(oldCosts)
print(oldDistances)
