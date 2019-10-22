import csv

data = []
pyes = 0
pno = 0
size = 0

with open("golf.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
        if(row[4] == 'yes'):
            pyes+=1
        else:
            pno+=1
        size+=1

# print data
print(data)

# read attributes to predict
print("Choose weather condition to predict Golf to be played or not")

predict = []
predict.append(input("Choose Outlook[sunny, overcast, rainy] from for:"))
predict.append(input("Choose Temperature[hot, mild, cool] from for:"))
predict.append(input("Choose Humidity[high, normal] from for:"))
predict.append(input("Choose Outlook[true, false] from for:"))

# count each attribute for playing and not playing
playing = [0,0,0,0]
not_playing = [0,0,0,0]
for i in range(len(data)):
    for j in range(len(predict)):
        if(predict[j] == data[i][j]):
            if(data[i][4]=='yes'):
                playing[j]+=1
            else:
                not_playing[j]+=1

# calculate probabilty
probability = [pyes/size, pno/size]   
for i in range(len(playing)):
    probability[0]*=(playing[i]/pyes)
    probability[1]*=(not_playing[i]/pno)

print(probability)
if probability.index(max(probability)) == 0:
    print("Playing")
else:
    print("Not Playing")
