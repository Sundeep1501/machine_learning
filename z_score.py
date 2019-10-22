from scipy.io import arff
import numpy
import math
data, meta = arff.loadarff(open('diabetes.arff.txt', 'r'))

nSamples = data.size
means = [0,0,0,0,0,0,0,0]

#find means for each attribute
for i in range(len(data)):
    sample = data[i]
    for j in range(len(means)):
        means[j] = sample[j] + means[j]

for j in range(len(means)):
        means[j] = means[j] / nSamples

print("Means")
print(means)
print()

#find standard deviation
sd = [0,0,0,0,0,0,0,0]
for i in range(len(data)):
    old = data[i]
    for j in range(len(means)):
        delta = old[j] - means[j]
        sd[j] = sd[j] + (delta * delta)

for j in range(len(means)):
    sd[j] = math.sqrt(sd[j]/nSamples)

print("Standard Deviations")
print(sd)
print()

for i in range(len(data)):
    sample = data[i]
    new = []
    for j in range(len(means)):
        old = sample[j]
        new.append((old-means[j]) / sd[j])
    print(new)
