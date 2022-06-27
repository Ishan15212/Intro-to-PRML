

import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import math
from scipy import stats

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datab1 = unpickle('cifar-10-batches-py/data_batch_1')
datab2 = unpickle('cifar-10-batches-py/data_batch_2')
datab3 = unpickle('cifar-10-batches-py/data_batch_3')
datab4 = unpickle('cifar-10-batches-py/data_batch_4')
datab5 = unpickle('cifar-10-batches-py/data_batch_5')
test_data = unpickle('cifar-10-batches-py/test_batch')

X = datab1["data"]
Y = datab1["labels"]
x_test_data = test_data["data"]
x_test_data = x_test_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
y_test_labels = test_data["labels"]
y_test_labels = np.array(y_test_labels)

labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

image_data = np.concatenate((datab1["data"],datab2["data"],datab3["data"],datab4["data"],datab5["data"]), axis=0)
image_labels = np.concatenate((datab1["labels"],datab2["labels"],datab3["labels"],datab4["labels"],datab5["labels"]), axis=0)
image_data = image_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

def class_acc(pred, gt):
    count = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            count = count + 1
    acc = (float(count)/float(len(pred)))*100
    print("{:.2f}".format(round(acc, 2)), '%')
    return round(acc,2)

def cifar10_color(X):
    mean_matrix = np.zeros((X.shape[0],3))

    for i in range(X.shape[0]):
        img = X[i]
        image_resize = resize(img, (1,1), mode='constant')

        r = image_resize[:, :, 0].reshape(1 * 1)
        g = image_resize[:, :, 1].reshape(1 * 1)
        b = image_resize[:, :, 2].reshape(1 * 1)

        mu_r = r.mean()
        mu_g = g.mean()
        mu_b = b.mean()
        mean_matrix[i, :] = (mu_r, mu_g, mu_b)
    return mean_matrix

def cifar10_NXN_color(X, N=2):
    mean = []
    for i in range(X.shape[0]):
        img = X[i]
        Im_re = resize(img, (N, N), mode= 'constant')
        r = Im_re[:, :, 0].reshape(N * N)
        g = Im_re[:, :, 1].reshape(N * N)
        b = Im_re[:, :, 2].reshape(N * N)

        mu_r = r
        mu_g = g
        mu_b = b
        mean.append([mu_r, mu_g, mu_b])
    mean = np.array(mean)
    return mean

    
def cifar_10_naivebayes_learn(Xp, Y):
    mean = {}
    for i in range(len(Y)):
        if Y[i] in mean:
            prev = mean[Y[i]]
            mean[Y[i]] = [(prev[0] + [Xp[i][0]]), (prev[1] + [Xp[i][1]]), (prev[2] + [Xp[i][2]])]
        else:
            mean[Y[i]] = [[Xp[i][0]], [Xp[i][1]], [Xp[i][2]]]
    
    mu = np.zeros((len(mean),3))
    sigma = np.zeros((len(mean),3))
    p = np.zeros((len(mean),1))
    for i in mean:
        mean_data = mean[i]
        mu[i] = [np.mean(mean_data[0]), np.mean(mean_data[1]), np.mean(mean_data[2])]
        sigma[i] = [np.var(mean_data[0]), np.var(mean_data[1]), np.var(mean_data[2])]
        p[i] = float(len(mean_data[0])) / float(len(Y))
    return mu,sigma,p

def norm_distribution(x, mean, sigma):
    var = (x - mean) ** 2
    denominator = math.sqrt(2*math.pi* sigma)
    numerator = math.exp((-0.5 * var)/sigma)
    return numerator/denominator

def cifar_10_classifier_naivebayes(x,mu,sigma,p):
    Xf = cifar10_color(x)
    labels = []
    new_probs = {}
    for i in range(x.shape[0]):
        for j in range(mu.shape[0]):
            nR = norm_distribution(Xf[i][0], mu[j][0], sigma[j][0])
            nG = norm_distribution(Xf[i][1], mu[j][1], sigma[j][1])
            nB = norm_distribution(Xf[i][2], mu[j][2], sigma[j][2])
            new_probs[j] = nR * nG * nB * p[j]
        key_max = max(new_probs, key= lambda x: new_probs[x])
        labels.append(key_max)
    return labels

def cifar_10_bayes_learn(Xf, Y, N):
    mean = {}
    image_size = N * N * 3
    for i in range(len(Y)):
        if Y[i] not in mean:
            mean[Y[i]] = [Xf[i].reshape(image_size)]                  
        else:
            prev = mean[Y[i]]
            mean[Y[i]] = np.append(prev, [Xf[i].reshape(N * N * 3)], axis=0)
    mu = np.zeros((len(mean), image_size))
    sigma = np.zeros((len(mean),image_size, image_size))
    p = np.zeros((len(mean),1))
    for i in mean:
        mean_data = mean[i]
        mu[i] = np.mean(mean_data, axis=0)
        sigma[i] = np.cov(np.transpose(mean_data))
        p[i] = float(len(mean_data[0])) / float(len(Y))
    return mu,sigma,p 
  
def cifar_10_classifier_bayes(x, mu, sigma, p, N):
    Xf = cifar10_NXN_color(x, N) #mean_matrix
    labels = []
    new_probs = {}
    for i in range(x.shape[0]):
        for j in range(mu.shape[0]):
            normal_d = stats.multivariate_normal(mean=mu[j], cov=sigma[j], allow_singular=True)
            new_probs[j] = normal_d.pdf(np.transpose(Xf[i].reshape(N * N * 3))) * p[j]
        key_max = max(new_probs, key= lambda x: new_probs[x])
        labels.append(key_max)
    return labels 

n = 4
acc_lib = []
for i in range(1,n+1):
    Xf_d = cifar10_NXN_color(image_data, i)
    mu, sigma, p = cifar_10_bayes_learn(Xf_d, image_labels, i)
    labels = cifar_10_classifier_bayes(x_test_data, mu, sigma, p, i)
    print('Accuracy of', i, "x", i)
    acc_lib.append(class_acc(labels, y_test_labels))
    print("---------------------------------------------------------")
    

i =[1,2,3,4]
plt.figure()
plt.plot(i,acc_lib,'-')
plt.plot(i,acc_lib,'*g')

plt.show()
    