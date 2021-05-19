#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:26:12 2019

@author: crystal
"""

from sklearn.externals import joblib ## replacement of pickle to carry large numpy arrays
import pickle
import sys

bnpyPath = '/Users/crystal/Documents/bnpy/bnpy'
sys.path.append(bnpyPath)
import bnpy
from bnpy.util.AnalyzeDP import * 
import numpy as np
import os
from six.moves import cPickle
# trueY = pickle.load(open('/Users/crystal/Documents/VaDE/latent_mnist.pkl', 'rb'))['y']
# fittedY = joblib.load('/Users/crystal/Documents/VaDE_results/fittedY.pkl')
# accResult = clusterAccuracy(trueY, fittedY)
from sklearn.utils import shuffle


import argparse
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-prop', action='store', type = float, dest='prop', default=0.6, \
                    help='the proportion of original data')
parser.add_argument('-rootPath', action='store', type = str, dest='rootPath', default='/Users/crystal/Documents/VaDE', \
                    help='root path to VaDE')
parser.add_argument('-useLocal', action='store_false', dest='useLocal', help='if use Local, rep environment variable will not be used')
import gzip



results = parser.parse_args()
prop = results.prop
rootPath = results.rootPath


def load_data(dataset, root_path, flatten=True, numbers=range(10)):
    path = os.path.join(os.path.join(root_path, 'dataset'), dataset)
    # path = 'dataset/'+dataset+'/'
    if dataset == 'mnist':
        path = os.path.join(path, 'mnist.pkl.gz')
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')
    
        if sys.version_info < (3,):
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")
        


        f.close()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        if flatten:
            x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
            x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        X = np.concatenate((x_train,x_test))
        if not flatten:
            X = np.expand_dims(X, axis=-1)
        Y = np.concatenate((y_train,y_test))

        if len(numbers) == 10:
            pass
        else:
            indices = []
            for number in numbers:
                indices += list(np.where(Y == number)[0])
            #indices = np.vstack(indices)
            X = X[indices]
            Y = Y[indices]
        
    if dataset == 'reuters10k':
        data=scio.loadmat(os.path.join(path,'reuters10k.mat'))
        X = data['X']
        Y = data['Y'].squeeze()
        
    if dataset == 'har':
        data=scio.loadmat(path+'HAR.mat')
        X=data['X']
        X=X.astype('float32')
        Y=data['Y']-1
        X=X[:10200]
        Y=Y[:10200]

    if dataset == 'stl10':
        with open('./dataset/stl10/X.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('./dataset/stl10/Y.pkl', 'rb') as f:
            Y = pickle.load(f)
            # here Y is one-hot, turn it back
            Y = np.argmax(Y, axis=1)

    return X,Y


## only load numbers in the list of numberList
numberList = [1, 3, 4, 5, 6, 7, 8, 9]
XSelect, YSelect = load_data('mnist', root_path = rootPath, flatten=True, numbers=numberList)
## randomly select 80% of the data and save it
## set the random seed when selecting the data so that the experimet is reproducible
np.random.RandomState(seed=1)
total_obs = len(YSelect)
nTrain = np.round(total_obs * prop)
## used as index for training data to get fitted model
train_ind = np.random.choice(total_obs, int(nTrain), replace=False)
## the rest of the data as test is to put into the new batch with data haven't emerged yet
Xtrain = XSelect[train_ind]
Ytrain = YSelect[train_ind]

test_ind = np.setdiff1d(np.arange(0, total_obs, 1), train_ind)
##################################################################3
## get all the 0 and 2s 
newNumberList1 = [0]
newNumberList2 = [2]
XNew1, YNew1 = load_data('mnist', root_path = rootPath, flatten=True, numbers=newNumberList1)
XNew2, YNew2 = load_data('mnist', root_path = rootPath, flatten=True, numbers=newNumberList2)


## merge half the data with index test_ind and half 0s first to be the first batch, 
## save the data
## merge half the data with index test_ind, the rest half of 0s and the 2s to be the second batch
## save the data

nObs1 = len(YNew1)
nObs2 = len(YNew2)

ind1 = np.random.choice(nObs1, int(np.round(0.5 * nObs1)), replace=False)
ind2 = np.random.choice(nObs2, int(np.round(0.5 * nObs2)), replace=False)

XNewSubset1 = XNew1[ind1, :]
YNewSubset1 = YNew1[ind1]

ind1Diff = np.setdiff1d(np.arange(0, nObs1, 1), ind1)
XNewSubset1otherHalf = XNew1[ind1Diff, :] 
YNewSubset1otherHalf = YNew1[ind1Diff]

XNewSubset2 = XNew2[ind2, :]
YNewSubset2 = YNew2[ind2]


test_total = len(test_ind)
firstHalf = int(np.round(test_total/2))
otherHalf = test_total - firstHalf

firstHalf_ind = np.random.choice(test_total, firstHalf, replace=False)
otherHalf_ind = np.random.choice(test_total, otherHalf, replace=False)

## merge first half of data
XTestFirst_ind  = test_ind[firstHalf_ind]
XTestSecond_ind = test_ind[otherHalf_ind]

XTestFirst = XSelect[XTestFirst_ind, :]
XTestSecond = XSelect[XTestSecond_ind, :]
YTestFirst = YSelect[XTestFirst_ind]
YTestSecond = YSelect[XTestSecond_ind]


XFirstBatch = np.concatenate((XTestFirst, XNewSubset1)) 
YFirstBatch = np.concatenate((YTestFirst, YNewSubset1))

XSecondBatch = np.concatenate((XTestSecond, XNewSubset1otherHalf, XNewSubset2))
YSecondBatch = np.concatenate((YTestSecond, YNewSubset1otherHalf, YNewSubset2))

## randomly shuffle each dataset 
XFirst, YFirst = shuffle(XFirstBatch, YFirstBatch, random_state=0)
XSecond, YSecond = shuffle(XSecondBatch, YSecondBatch, random_state=0)

## pickle the data
with open('../newCluster/Xtrain.pkl', 'wb') as f:
    pickle.dump(Xtrain, f)
with open('../newCluster/Ytrain.pkl', 'wb') as f:
    pickle.dump(Ytrain, f)
    
with open('../newCluster/XFirstBatch.pkl', 'wb') as f:
    pickle.dump(XFirst, f)
with open('../newCluster/YFirstBatch.pkl', 'wb') as f:
    pickle.dump(YFirst, f)

with open('../newCluster/XSecondBatch.pkl', 'wb') as f:
    pickle.dump(XSecond, f)

with open('../newCluster/YSecondBatch.pkl', 'wb') as f:
    pickle.dump(YSecond, f)




