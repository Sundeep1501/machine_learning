from scipy.io import arff
import numpy
data, meta = arff.loadarff(open('diabetes.arff.txt', 'r'))

nSamples = data.size
mins = [0,0,0,0,0,0,0,0]
maxs = [0,0,0,0,0,0,0,0]

#init min max with first sample
firstSample = data[0]
for i in range(len(mins)):
    mins[i] = maxs[i] = firstSample[i]

#compare and find min and max from all samples
for i in range(len(data)):
    sample = data[i]
    for j in range(len(mins)):
        if(mins[j] > sample[j]):
            mins[j] = sample[j]
        if(maxs[j] < sample[j]):
            maxs[j] = sample[j]

for i in range(len(data)):
    old = data[i]
    new = []
    for j in range(len(mins)):
        new.append((old[j] - mins[j])/(maxs[j] - mins[j]))
    print(new)
