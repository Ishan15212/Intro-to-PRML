

import pickle
import numpy as np
from numpy.random import randint
from random import random
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

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

X_train =  np.array(X_all.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("int32"));
Y_train =  np.array(Y_all);

datadict = unpickle('cifar-10-batches-py/test_batch');
X_test = datadict["data"];
Y_test = datadict["labels"];
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32");




model= tf.keras.models.Sequential();

model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(32,32,3), strides=(1,1), padding="valid", activation=tf.nn.relu));
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"));

model.add(tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation=tf.nn.relu));
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"));

model.add(tf.keras.layers.Flatten());
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu));
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax));



opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#/******* fit model ********/
history=model.fit(X_train,Y_train,epochs=80,batch_size=64);
print(model.summary());

predictions=model.predict([X_test]);
plt.imshow(X_test[1]);
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"];
print("real Label is "+str(label_names[Y_test[1]]))
print("Predicted Label is "+str(label_names[np.argmax(predictions[1])]))
print("prediction is "+str(np.argmax(predictions[1])));


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
