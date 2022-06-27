import pickle
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from random import random
import math

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred,gt):
    errorSum=0;
    accuracyOfModal=0;
    for i in range(0,len(gt)):
        if gt[i]!=pred[i]:
            #print("gt is",gt[i],"pred is",pred[i])
            #print("gt is {gt[i]} and pred is {pred[i]}")
            errorSum=errorSum+1;
    accuracyOfModal=(pred.shape[0]-errorSum)/(pred.shape[0])
    accuracyPercentage=accuracyOfModal*100
    return accuracyPercentage

def cifar10_classifier_random(x):
    y=randint(0, 9, x.shape[0])
    #y=np.full(x.shape[0], 5)
    return y

def cifar10_classier_1nn(X,trdata,trlabels,predY):
    #minDistance=X.shape(32, 32, 3);
    print(trdata.shape)
    print(X.shape)
    #predLabel=np.full(10000, 0)
    #predLabel=predY;
    predLabel=predY[0:100];
    #minDistance=(math.sqrt(sum(pow((X[0]-trdata[0]),2))))+0.0001
    #minDistance=10000000000
    minDistance=np.full(100, 1000000000)
    imageCount=0;
    labelIndex=0;
    comparedImageCount=0;
    euclidentDistance=0;
    #minDistance=minDistance.reshape(3, 32, 32).transpose(1,2,0).astype("uint8")
    #minDistance=minDistance.reshape(32, 32).transpose(0,1).astype("uint8")
    #euclidentDistance=sum(sum(sum(sum(pow((X-trdata),2)))));
    #for i in X:
    for i in X[0:100,:]:
        #minDistance[i]=10000000000
        euclidentDistance=0;
        comparedImageCount=0;
        #labelIndex=0
        #for j in range(trdata.shape[0]):
        #compare with only 1000 images
        for j in trdata[0:10000,:]:
            #print("comparing the image "+str(j))
            #euclidentDistance=sum(pow((X[i]-trdata[j]),2))
            #euclidentDistance=abs(sum(sum((X[i]-trdata[j]))))
            euclidentDistance=abs(sum(sum((X[i]-trdata[j]))))
            #print(euclidentDistance)
            #euclidentDistance= euclidentDistance[0]
            #euclidentDistance=(math.sqrt(sum(pow((X[i]-trdata[j]),2))))
            #print(euclidentDistance)
            if euclidentDistance < minDistance[imageCount]:
                #minDistance=euclidentDistance
                minDistance[imageCount]=euclidentDistance
                print(str(minDistance[imageCount])+" minDistance")
                labelIndex=trlabels[comparedImageCount];
                print(str(labelIndex)+" labelIndex")
                print("-----------------------------------------------------------------")
                comparedImageCount=comparedImageCount+1;
                if minDistance[imageCount] == 0.0:
                  break
        predLabel[imageCount]=labelIndex
        #print(str(imageCount)+"i")
        print(predLabel)
        print("/********************************** new image "+str(imageCount)+" *******************************************/")
        #print(str(imageCount)+" image")
        imageCount=imageCount+1;
    #print(minDistance.shape)
    #print(euclidentDistance.shape)
    #comparision=minDistance<euclidentDistance
    # if comparision.all():
    #     minDistance=euclidentDistance
    #minDistance=np.full(, 5)
    
    #minDistance=min(euclidentDistance,minDistance)
    #print(trlabels)
    print(predLabel)
    return predLabel,minDistance
#datadict = unpickle('/content/drive/MyDrive/cifar-10-batches-py/data_batch_1')
datadict = unpickle('/content/drive/MyDrive/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('/content/drive/MyDrive/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
#print(X.shape)
#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#X = X.reshape(10000, 3, 32, 32).transpose(0,1,2,3).astype("uint8")
Y = np.array(Y)

# for i in range(X.shape[0]):
#     # Show some images randomly
#     if random() > 0.999:
#         plt.figure(1);
#         plt.clf()
#         plt.imshow(X[i])
#         plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")l
#         plt.pause(1)

#predY=cifar10_classifier_random(X)
predY=np.full(X.shape[0], 0)
minDistance=np.full(10, 0)
#batchArr=[2,3,4,5]
batchArr=[3]
for i in batchArr:
    datadict = unpickle('/content/drive/MyDrive/cifar-10-batches-py/data_batch_'+str(i))
    trdata = datadict["data"]
    trlabels = datadict["labels"]
    predictedY,minDistance=cifar10_classier_1nn(X,trdata,trlabels,predY)
    #trdata = trdata.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#datadict = unpickle('cifar-10-batches-py/data_batch_'+str(i))
#datadict = unpickle('cifar-10-batches-py/data_batch_2')
#trdata = datadict["data"]
#trlabels = datadict["labels"] 
#trdata = trdata.reshape(10000, 3, 32, 32).transpose(0,1,2,3).astype("uint8")

predY=cifar10_classifier_random(X);
randomAccuracy=class_acc(predY,Y)
Y=Y[0:100];
print("Predicted Y set is Given below")
print(predY)
print("Actual Y set is Given below")
print(Y)
predictedY=predictedY[0:100]
accuracyPercentage=class_acc(predictedY,Y)
#print(predY)
print(minDistance)
print("Random accuracy "+str(randomAccuracy))
print("Total Accuracy "+str(accuracyPercentage))