import pickle
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from random import random
import math
from scipy.stats import norm
from skimage.transform import resize

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def cifar10_color(X_for_test): 

    X=X_for_test;
    Xp = np.zeros((X.shape[0],3))

    for i in range(X.shape[0]):
        I = X[i]
        Iresize = resize(I,(1,1))
        Xp[i] = Iresize

        
    return Xp


def calculateParams(X,Y):
    meanMatrix = np.array([0,0,0], dtype = np.int32)
    varMatrix = np.array([0,0,0], dtype = np.int32)
    priorMatrix = np.array([0], dtype = np.int32)
    for classNumber in range(0,10):
        filter_Y = Y == classNumber;
        filtered_X = X[filter_Y];
        prior=(filtered_X.shape[0]*100)/(X.shape[0]);
        getMean = np.mean(filtered_X,axis=0)
        getVar = np.var(filtered_X, axis=0)
        meanMatrix=np.vstack((meanMatrix,getMean))
        varMatrix=np.vstack((varMatrix,getVar))
        priorMatrix=np.vstack((priorMatrix,prior))
    meanMatrix = np.delete(meanMatrix, 0, 0)
    varMatrix = np.delete(varMatrix, 0, 0)
    priorMatrix = np.delete(priorMatrix, 0, 0)
    return meanMatrix,varMatrix,priorMatrix

def normal_dist(x,mean,sd):
    prob_density = (1/np.sqrt(2*np.pi*sd))*(np.exp(-0.5*((x-mean)**2)/(1*sd)))
    print("--------- now probab density ----------------")
    print(prob_density)
    return prob_density

def cifar10classiernaivebayes(x,mu,sigma,p):
    sumND = np.zeros((1,x.shape[0]), dtype = np.int32)
    allNumerator = np.array(sumND, dtype = np.int32)
    getDenominator = np.array(sumND, dtype = np.int32);
    getND=np.array([], dtype = np.int32);

    for classNumber in range(0,10):
        getND=normal_dist(x,mu[classNumber],sigma[classNumber]);
        getNumerator=(getND[:,0]*getND[:,1]*getND[:,2])*p[classNumber];
        sumND=sumND+(np.transpose(getNumerator));
        allNumerator=np.vstack((allNumerator,getNumerator))
    allNumerator = np.transpose(np.delete(allNumerator, 0, 0))
    getDenominator=np.transpose(allNumerator)*p
    getDenominator=(np.transpose(getDenominator).sum(axis=1))
    probab=np.transpose(allNumerator)/getDenominator;
    probab=np.transpose(probab)
    estimatedClass=probab.argmax(axis=1)
    return estimatedClass

def class_acc(pred,gt):
    errorSum=0;
    accuracyOfModal=0;
    for i in range(0,len(gt)):
        if gt[i]!=pred[i]:
            errorSum=errorSum+1;
    accuracyOfModal=(pred.shape[0]-errorSum)/(pred.shape[0])
    accuracyPercentage=accuracyOfModal*100
    return accuracyPercentage


a=[1,2,3,4,5];
X_all = np.array([], dtype = np.int32)
Y_all = np.array([], dtype = np.int32)
for i in a:
  datadict = unpickle('cifar-10-batches-py/data_batch_'+str(i));
  X = datadict["data"];
  Y = datadict["labels"];
  X_all=np.append(X_all,X);
  Y_all=np.append(Y_all,Y);

X_all = X_all.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("int32")

X=cifar10_color(X_all);
specificClass=0;
meanMatrix,varMatrix,priorMatrix=calculateParams(X,Y_all)

datadict = unpickle('cifar-10-batches-py/test_batch');
X_test = datadict["data"];
Y_test = datadict["labels"];
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
#print(X_test)
X_test=cifar10_color(X_test);
#print(X_test)
Y_test=np.array(Y_test, dtype = np.int32);
estimatedY=cifar10classiernaivebayes(X_test,meanMatrix,varMatrix,priorMatrix);
accuracy=class_acc(estimatedY,Y_test)
print("Accuracy Of Task 1 is: "+str(accuracy))



