# %%
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Flatten
from keras.models import Model, load_model, model_from_json
from keras import backend as K

from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import os
import sys
import argparse
import numpy as np

import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf
from sklearn.externals import joblib  ## replacement of pickle to carry large numpy arrays
import pickle

# This is for not eating up the whole RAM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
gpu_fraction = 0.1
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)
# set_session(tf.Session(config=config))
# os.chdir("/home/tingting/code/")
os.chdir("/Users/crystal/Documents/")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-bnpyPath', action='store', type=str, dest='bnpyPath', default='./bnpy/', \
                    help='path to bnpy code repo')
# parser.add_argument('-outputPath', action='store', type = str, dest='outputPath', default='../Documents/VaDE_results/', \
#                    help='path to output')
parser.add_argument('-outputPath', action='store', type=str, dest='outputPath',
                    default='/Users/crystal/Documents/VaDE_results/', \
                    help='path to output')
parser.add_argument('-rootPath', action='store', type=str, dest='rootPath', default='./VaDE', \
                    help='root path to VaDE')
parser.add_argument('-conv', action='store_true', \
                    help='using convolutional autoencoder or not')
parser.add_argument('-logFile', action='store_true', dest='logFile', help='if logfile exists, save the log file to txt')
parser.add_argument('-useLocal', action='store_true', dest='useLocal',
                    help='if use Local, rep environment variable will not be used')
## add argument for the maximum number of clusters in DP
parser.add_argument('-Kmax', action='store', type=int, dest='Kmax', default=50,
                    help='the maximum number of clusters in DPMM')
## parse data set option as an argument
parser.add_argument('-dataset', action='store', type=str, dest='dataset', default='mnist',
                    help='the options can be mnist,reuters10k and har')
parser.add_argument('-epoch', action='store', type=int, dest='epoch', default=1, help='The number of epochs')
parser.add_argument('-batch_iter', action='store', type=int, dest='batch_iter', default=2,
                    help='The number of updates in SGVB')
parser.add_argument('-scale', action='store', type=float, dest='scale', default=0.005,
                    help='the scale parameter in the loss function')
parser.add_argument('-batchsize', action='store', type=int, dest='batchsize', default=1000,
                    help='the default batch size when training neural network')
parser.add_argument('-nBatch', action='store', type=int, dest='nBatch', default=1, help='number of batches in DP')
parser.add_argument('-sf', action='store', type=float, dest='sf', default=0.1,
                    help='the prior diagonal covariance matrix for Normal mixture in DP')
parser.add_argument('-gamma0', action='store', type=float, dest='gamma0', default=5.0,
                    help='hyperparameters for DP in Beta dist')
parser.add_argument('-gamma1', action='store', type=float, dest='gamma1', default=1.0,
                    help='hyperparameters for DP in Beta dist')
parser.add_argument('-rep', action='store', type=int, dest='rep', default=1, help='add replication number as argument')
parser.add_argument('-nLap', action='store', type=int, dest='nLap', default=500, help='the number of laps in DP')
parser.add_argument('-threshold', action='store', type=float, dest='threshold', default=0.88, help='stopping criteria')
parser.add_argument('-useNewPretrained', action='store_true', dest='useNewPretrained',
                    help='Indicator about using new pretrained weights')
parser.add_argument('-taskID', action='store', type=int, dest='taskID', default=1,
                    help='use taskID to random seed for bnpy')
parser.add_argument('-learningRate', action='store', type=float, dest='lr', default=1e-5,
                    help='the learning rate in adam_nn')

results = parser.parse_args()
results.useLocal = True
results.logFile = True
if results.useLocal:
    rep = results.rep
else:
    rep = os.environ["rep"]
    rep = int(float(rep))

bnpyPath = results.bnpyPath
sys.path.append(bnpyPath)
subdir = os.path.join(bnpyPath, 'bnpy')
sys.path.append(subdir)
outputPath = results.outputPath
from GenImageUtil import *

if not os.path.exists(outputPath):
    os.mkdir(outputPath)

root_path = results.rootPath
sys.path.append(root_path)

online_path = os.path.join(outputPath, 'singledigit2taskid4rep4')
isOnlineRead = True
isOnlineWrite = True
subset=True

Kmax = results.Kmax
dataset = results.dataset
epoch = results.epoch
batch_iter = results.batch_iter
scale = results.scale
batchsize = results.batchsize
lr_input = results.lr
## DP hyper-parameters
sf = results.sf
gamma0 = results.gamma0
gamma1 = results.gamma1
threshold = results.threshold
nBatch = results.nBatch
taskID = results.taskID

from OrganizeResultUtil import createOutputFolderName, createFullOutputFileName

## Rep is useful when running the same experiment multiple times to obtain a standard error

flatten = True
if results.conv:
    flatten = False


number = 9
## specify full output path
outputPathNew = os.path.join(outputPath, 'singledigit2taskid4rep4')
fullOutputPath = createOutputFolderName(outputPathNew, Kmax, dataset, epoch, batch_iter, scale, batchsize, rep, sf)
## name log file and write console output to log.txt
logFilePath = os.path.join(outputPathNew, str(number))
logFileName = os.path.join(logFilePath, 'log.txt')

if results.logFile:
    sys.stdout = open(logFileName, 'w')

#############################################
import DP as DP
from bnpy.util.AnalyzeDP import *
from bnpy.data.XData import XData
import pickle

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def load_data(dataset, root_path, flatten=True, numbers=range(10), N=10000, percentage=0.2, subset=True):
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

        X = np.concatenate((x_train, x_test))
        if not flatten:
            X = np.expand_dims(X, axis=-1)
        Y = np.concatenate((y_train, y_test))

        if len(numbers) == 10:
            pass
        else:
            indices = []
            if subset is True:
                ## take a subset of the samples out of all the numbers
                ## take a percentage of the new cluster based on "number"
                numberNew = numbers[-1]
                for numberDigit in numbers:
                    ## number as the last element of numbers
                    if not numberDigit == numberNew:
                        indices += list(np.where(Y == numberDigit)[0])

                np.random.RandomState(seed=1)
                if percentage <1.0:
                    ## take a subset of the old samples

                    old_ind = np.random.choice(indices, int(N*(1-percentage)), replace=False)
                    indicesNew = []
                    indicesNew += list(np.where(Y == numberNew)[0])
                    new_ind = np.random.choice(indicesNew, int(N * percentage), replace=True)
                    total_ind = list(old_ind) + list(new_ind)
                else:
                    indicesNew = []
                    indicesNew += list(np.where(Y == numberNew)[0])
                    ## sample N samples
                    total_ind = np.random.choice(indicesNew, int(N), replace=True)

                X = X[total_ind]
                Y = Y[total_ind]
                # print(round(len(new_ind)/len(total_ind)))

            else:
                for numberDigit in numbers:
                    indices += list(np.where(Y == numberDigit)[0])
                # indices = np.vstack(indices)
                X = X[indices]
                Y = Y[indices]


    if dataset == 'reuters10k':
        data = scio.loadmat(os.path.join(path, 'reuters10k.mat'))
        X = data['X']
        Y = data['Y'].squeeze()

    if dataset == 'har':
        data = scio.loadmat(path + 'HAR.mat')
        X = data['X']
        X = X.astype('float32')
        Y = data['Y'] - 1
        X = X[:10200]
        Y = Y[:10200]

    if dataset == 'stl10':
        with open('./dataset/stl10/X.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('./dataset/stl10/Y.pkl', 'rb') as f:
            Y = pickle.load(f)
            # here Y is one-hot, turn it back
            Y = np.argmax(Y, axis=1)

    return X, Y


def config_init(dataset):
    # original_dim,epoches,n_centroid,lr_nn,lr_gmm,decay_n,decay_nn,decay_gmm,alpha,datatype
    if dataset == 'mnist':
        # return 784,3000,10,0.002,0.002,10,0.9,0.9,1,'sigmoid'
        return 784, 3000, 10, 2e-4, 0.002, 10, 0.9, 0.9, 1, 'sigmoid'
    if dataset == 'reuters10k':
        return 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1, 'linear'
    if dataset == 'har':
        return 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5, 'linear'
    if dataset == 'stl10':
        return 2048, 10, 10, 0.002, 0.002, 10, 0.9, 0.9, 1, 'linear'


def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)

    return loss

def load_pretrain_online_weights(vade, online_path, number):
    # OnlineModelFolder = os.path.join(online_path, str(number-1))
    OnlineModelFolder = os.path.join(online_path, str(number - 2))
    OnlineModelName = os.path.join(OnlineModelFolder, 'vade_DP_model.json')
    ae = model_from_json(open(OnlineModelName).read())

    OnlineWeightsName = os.path.join(OnlineModelFolder, 'vade_DP_weights.h5')
    vade.load_weights(OnlineWeightsName)

    #vade.layers[1].set_weights(ae.layers[0].get_weights())
    #vade.layers[2].set_weights(ae.layers[1].get_weights())
    #vade.layers[3].set_weights(ae.layers[2].get_weights())
    #vade.layers[4].set_weights(ae.layers[3].get_weights())

    #vade.layers[-1].set_weights(ae.layers[-1].get_weights())
    #vade.layers[-2].set_weights(ae.layers[-2].get_weights())
    #vade.layers[-3].set_weights(ae.layers[-3].get_weights())
    #vade.layers[-4].set_weights(ae.layers[-4].get_weights())

    return vade


def load_pretrain_weights(vade, root_path, dataset):
    if dataset == 'stl10':
        dataset += '_supervised'
    if dataset == 'reuters10k':
        dataset += '_supervised'
    ## The following lines are commented out since the results are not good, thus use the pretrained weights provided by the original author
    if dataset == 'mnist':
        dataset += '_supervised'
    path = os.path.join(root_path, 'pretrain_weights')
    filename = 'ae_' + dataset + '.json'
    fullFileName = os.path.join(path, filename)
    ae = model_from_json(open(fullFileName).read())
    # ae = model_from_json(open('pretrain_weights/ae_'+dataset+'.json').read())
    weightFileName = 'ae_' + dataset + '_weights.h5'
    weightFullFileName = os.path.join(path, weightFileName)
    ae.load_weights(weightFullFileName)

    if 'stl10' not in dataset and 'reuters10k' not in dataset and 'mnist' not in dataset:
        # ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
        # if results.useNewPretrained:
        vade.layers[1].set_weights(ae.layers[0].get_weights())
        vade.layers[2].set_weights(ae.layers[1].get_weights())
        vade.layers[3].set_weights(ae.layers[2].get_weights())
        vade.layers[4].set_weights(ae.layers[3].get_weights())

        vade.layers[-1].set_weights(ae.layers[-1].get_weights())
        vade.layers[-2].set_weights(ae.layers[-2].get_weights())
        vade.layers[-3].set_weights(ae.layers[-3].get_weights())
        vade.layers[-4].set_weights(ae.layers[-4].get_weights())
    else:
        vade.layers[1].set_weights(ae.layers[1].get_weights())
        vade.layers[2].set_weights(ae.layers[2].get_weights())
        vade.layers[3].set_weights(ae.layers[3].get_weights())
        vade.layers[4].set_weights(ae.layers[4].get_weights())
        vade.layers[-1].set_weights(ae.layers[-2].get_weights())
        vade.layers[-2].set_weights(ae.layers[-3].get_weights())
        vade.layers[-3].set_weights(ae.layers[-4].get_weights())
        vade.layers[-4].set_weights(ae.layers[-5].get_weights())

    return vade


def load_pretrain_cnn_encoder(encoder, root_path, model='cnn_classifier.05-0.02.hdf5'):
    print("Loading Pretrained Weights for CNN-VAE-Encoder!")
    path = os.path.join(root_path, 'conv_classifier_pre_weights', model)
    # layer cnn: 1, 3, 5, dense:8
    pre_encoder = load_model(path)
    for lid in [1, 3, 5, 8]:
        encoder.layers[lid].set_weights(pre_encoder.layers[lid].get_weights())
    return encoder


def load_pretrain_cnn_vae(vade, root_path, model='vae_cnn_mnist_semi_supervised'):
    with open(os.path.join(root_path, 'conv_vae_pre_weights', model + '.json')) as f:
        pre_vae = model_from_json(f.read())
    pre_vae.load_weights(os.path.join(root_path, 'conv_vae_pre_weights', model + '.weights'))
    for lid in [1, 3, 5, 8]:
        vade.layers[lid].set_weights(pre_vae.layers[lid].get_weights())
    for lid in [-1, -3, -5, -7, -9]:
        vade.layers[lid].set_weights(pre_vae.layers[lid - 1].get_weights())
    return vade


'''def elbo_nn(DPParam):
    #gamma = DPParam['LPMtx']
    #N = DPParam['Nvec']
    #m = DPParam['m']
    #W = DPParam['W']
    #v = DPParam['nu']
    #k = v.shape[0]
    def loss(x, x_decoded_mean):
        N = tf.convert_to_tensor(DPParam, dtype=tf.float32)
        loss_=alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean) + \
        -0.5 * K.sum(z_log_var, axis = -1) + N
        return loss_

        # line 93 term
    return loss'''


# gamma: 'LPMtx' (batch_size, # of cluster)
# N : 'Nvec' (# of cluster, )
# m : 'm' (# of cluster, latent_dim)
# W : 'B' (# of cluster, latent_dim, latent_dim)
# v: 'nu' (# of cluster)
# def loss_full_DP(x, x_decoded_mean):
## given z_mean, calculate the new ELBO in DP
#    model = DPParam['model']
## transform z_mean as tensor object into a python numpy array
#    z_mean_np = tf.keras.backend.eval(z_mean)
## transform the numpy array as XData type requrired by bnpy
#    z_mean_xdata = XData(z_mean_np,dtype='auto')

## get sufficient statistics
#    LP = model.calc_local_params(z_mean_xdata)
#    SS = model.get_global_suff_stats(z_mean, LP, doPrecompEntropy=1)
#   elbo = tf.convert_to_tensor(model.calc_evidence(z_mean_xdata, SS, LP), dtype=tf.float32)

#    loss_ = alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean) - elbo

#    ELBO = tf.convert_to_tensor(DPParam['elbo'], dtype = tf.float32)
#    loss_= alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean) - ELBO
#   loss = K.sum(loss_, axis = 0)
#    return loss


def loss(x, x_decoded_mean):
    # N = tf.convert_to_tensor(DPParam, dtype=tf.float32)

    gamma = tf.convert_to_tensor(DPParam['LPMtx'], dtype=tf.float32)
    N = tf.convert_to_tensor(DPParam['Nvec'], dtype=tf.float32)
    m = tf.convert_to_tensor(DPParam['m'], dtype=tf.float32)
    W = tf.convert_to_tensor(DPParam['B'], dtype=tf.float32)
    v = tf.convert_to_tensor(DPParam['nu'], dtype=tf.float32)

    num_cluster = N.shape[0]
    z_mean_1_last = tf.expand_dims(z_mean, -1)  # bs, latent_dim, 1
    z_mean_1_mid = tf.expand_dims(z_mean, 1)  # bs, 1, latent_dim

    for k in range(num_cluster):
        gamma_k_rep = tf.squeeze(K.repeat(tf.expand_dims(gamma[:, k], -1), latent_dim))
        z_k_bar = 1 / N[k] * K.sum(tf.multiply(gamma_k_rep, z_mean), axis=0)  # (latent_dim, )
        z_k_bar_batch = tf.squeeze(K.repeat(tf.expand_dims(z_k_bar, 0), batch_size))
        # tf.transpose(z_k_bar_batch, perm=[1, 0])
        z_k_bar_batch_1_last = tf.expand_dims(z_k_bar_batch, -1)  # bs, latent_dim, 1
        z_k_bar_batch_1_mid = tf.expand_dims(z_k_bar_batch, 1)  # bs, 1, latent_dim

        # TODO:!
        S_k = 1 / N[k] * K.sum(
            K.batch_dot(tf.multiply(tf.expand_dims(gamma_k_rep, -1), (z_mean_1_last - z_k_bar_batch_1_last)),
                        z_mean_1_mid - z_k_bar_batch_1_mid), axis=0)  # (latent_dim, latent_dim)
        temp = tf.linalg.trace(tf.linalg.solve(W[k], S_k))
        temp2 = tf.matmul(tf.expand_dims((z_k_bar - m[k]), 0), tf.linalg.inv(W[k]))
        temp3 = tf.squeeze(tf.matmul(temp2, tf.expand_dims((z_k_bar - m[k]), -1)))
        if k == 0:
            e = 0.5 * N[k] * (v[k] * (temp + temp3))
        else:
            e += 0.5 * N[k] * (v[k] * (temp + temp3))

    loss_ = alpha * original_dim * objectives.mean_squared_error(K.flatten(x),
                                                                 K.flatten(x_decoded_mean)) - scale * 0.5 * K.sum(
        (z_log_var + 1), axis=-1)
    loss_ = K.sum(loss_, axis=0) + e
    # loss = K.sum(loss_, axis = 0)
    # for i in range(5):
    #    loss_ += N

    # return loss_
    return loss_


def cnn_loss(x, x_decoded_mean):
    # N = tf.convert_to_tensor(DPParam, dtype=tf.float32)

    gamma = tf.convert_to_tensor(DPParam['LPMtx'], dtype=tf.float32)
    N = tf.convert_to_tensor(DPParam['Nvec'], dtype=tf.float32)
    m = tf.convert_to_tensor(DPParam['m'], dtype=tf.float32)
    W = tf.convert_to_tensor(DPParam['B'], dtype=tf.float32)
    v = tf.convert_to_tensor(DPParam['nu'], dtype=tf.float32)

    num_cluster = N.shape[0]
    z_mean_1_last = tf.expand_dims(z_mean, -1)  # bs, latent_dim, 1
    z_mean_1_mid = tf.expand_dims(z_mean, 1)  # bs, 1, latent_dim

    for k in range(num_cluster):
        gamma_k_rep = tf.squeeze(K.repeat(tf.expand_dims(gamma[:, k], -1), latent_dim))
        z_k_bar = 1 / N[k] * K.sum(tf.multiply(gamma_k_rep, z_mean), axis=0)  # (latent_dim, )
        z_k_bar_batch = tf.squeeze(K.repeat(tf.expand_dims(z_k_bar, 0), batch_size))
        # tf.transpose(z_k_bar_batch, perm=[1, 0])
        z_k_bar_batch_1_last = tf.expand_dims(z_k_bar_batch, -1)  # bs, latent_dim, 1
        z_k_bar_batch_1_mid = tf.expand_dims(z_k_bar_batch, 1)  # bs, 1, latent_dim

        # TODO:!
        S_k = 1 / N[k] * K.sum(
            K.batch_dot(tf.multiply(tf.expand_dims(gamma_k_rep, -1), (z_mean_1_last - z_k_bar_batch_1_last)),
                        z_mean_1_mid - z_k_bar_batch_1_mid), axis=0)  # (latent_dim, latent_dim)
        temp = tf.linalg.trace(tf.linalg.solve(W[k], S_k))
        temp2 = tf.matmul(tf.expand_dims((z_k_bar - m[k]), 0), tf.linalg.inv(W[k]))
        temp3 = tf.squeeze(tf.matmul(temp2, tf.expand_dims((z_k_bar - m[k]), -1)))
        if k == 0:
            e = 0.5 * N[k] * (v[k] * (temp + temp3))
        else:
            e += 0.5 * N[k] * (v[k] * (temp + temp3))

    loss_ = alpha * original_dim * objectives.mean_squared_error(K.flatten(x),
                                                                 K.flatten(x_decoded_mean)) - scale * K.sum(
        (z_log_var + 1), axis=-1)
    loss_ = K.sum(loss_, axis=0) + e
    # loss = K.sum(loss_, axis = 0)
    # for i in range(5):
    #    loss_ += N

    # return loss_
    return loss_


# dataset = 'reuters10k'
# db = sys.argv[1]
# if db in ['mnist','reuters10k','har']:
#    dataset = db
print('training on: ' + dataset)
ispretrain = True
batch_size = batchsize
latent_dim = 10
intermediate_dim = [500, 500, 2000]
# theano.config.floatX='float32'
accuracy = []
X, Y = load_data(dataset, root_path, flatten,  numbers=np.array([0, 1]), percentage = 0.5)

original_dim, epoches, n_centroid, lr_nn, lr_gmm, decay_n, decay_nn, decay_gmm, alpha, datatype = config_init(dataset)
vade_ini, encoder, decoder = get_models(model_flag='dense', batch_size=128, original_dim=784, latent_dim=10, intermediate_dim=[500, 500, 2000])
global DPParam

if isOnlineRead:
    online_full_path = os.path.join(online_path, str(number-2))
    DPParam_path = os.path.join(online_full_path, 'DPParam.pkl')
    DPModel_path = os.path.join(online_full_path, 'dp_model.pkl')
    DPInfo_path = os.path.join(online_full_path, 'dp_info_dict.pkl')
    with open(DPParam_path, 'rb') as f:
        DPParam = joblib.load(f)
    with open(DPModel_path, 'rb') as f:
        dp_model = joblib.load(f)
    with open(DPInfo_path, 'rb') as f:
        dp_info_dict = joblib.load(f)

if flatten:
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation=datatype)(h_decoded)

    sample_output = Model(x, z_mean)

    vade = Model(x, x_decoded_mean)
    if ispretrain == True:
        vade = load_pretrain_weights(vade, root_path, dataset)
    if ispretrain == True and isOnlineRead == True:
        vade = load_pretrain_online_weights(vade, online_path, number)
        # Read DP
        if True:
            m, W, nu, beta = obtainDPParam(DPParam)
            N = 10000
            X_single, Y_single = load_data(dataset, root_path, flatten=flatten, numbers=[(number-1), number], N=N, percentage=0.5)
            # 2000, 784; 2000,
            # new_load_data
            _, decoder = load_pretrain_vade_weights(encoder, decoder, vade)
            X_gen_list = [X_single]
            for nc in range(len(m)):
                mean = m[nc]
                var = W[nc] * 1 / float(nu[nc])
                z_sample = multivariate_normal(mean, var, int(N/3))
                X_generated = decoder.predict(z_sample)
                X_gen_list.append(X_generated)
            X = np.concatenate(X_gen_list, axis=0)
            Y_generated = -1 * np.ones((X_gen_list[1].shape[0] * (len(X_gen_list)-1)))
            Y = np.concatenate([Y_single, Y_generated], axis=0)

else:  # use CNN
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    # channel merge
    # x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # build decoder model
    # for generative model
    # latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(z)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # constructing several models
    sample_output = Model(input_img, z, name='encoder')
    # decoder = Model(latent_inputs, decoded, name='decoder')

    # decoded_for_vade = decoder(sample_output(input_img))
    vade = Model(input_img, decoded, name='vade')

    vade.summary()
    sample_output.summary()
    # decoder.summary()

    if ispretrain == True:
        # sample_output = load_pretrain_cnn_encoder(sample_output, root_path)
        # TODO:
        print("*********Loading Pretrained Weights for MNIST-CNN********")
        vade = load_pretrain_cnn_vae(vade, root_path)

num_of_exp = X.shape[0]

num_of_epoch = epoch
num_of_data_stream = int(num_of_exp / batch_size)

if 'reuters10k' in dataset or 'stl10' in dataset or results.conv:
    adam_nn = Adam(lr=lr_nn, epsilon=1e-5, decay=0.1)
if 'mnist' in dataset and not results.conv:
    adam_nn = Adam(lr=lr_input, epsilon=1e-5, decay=0.1)

# %%
global newinitname

if not flatten:
    print("Pretraining VaDE first!")
    vade.compile(optimizer='adadelta', loss='binary_crossentropy')
    vade.fit(X, X, epochs=2, batch_size=batch_size, validation_data=(X, X), shuffle=True)

gamma1 = 1.0
gamma0 = 5.0

dp_model = None
dp_info_dict = None

id_list = np.arange(num_of_exp)
np.random.shuffle(id_list)

for data_stream_iter in range(num_of_data_stream):
    print("The current data stream is stream: {}".format(data_stream_iter))
    indices = id_list[data_stream_iter * batch_size:(data_stream_iter + 1) * batch_size]
    x_batch = X[indices, :]
    z_batch = sample_output.predict_on_batch(x_batch)

    for iteration in range(num_of_epoch):
        if iteration ==0 and data_stream_iter==0:
            newinitname = 'randexamples'
            if dataset == 'reuters10k':
                DPObj = DP.DP(output_path=fullOutputPath, initname=newinitname, gamma1=gamma1, gamma0=gamma0, Kmax=Kmax,
                              sf=sf, nBatch=nBatch, taskID=taskID)
            else:
                DPObj = DP.DP(output_path=fullOutputPath, initname=newinitname, gamma1=gamma1, gamma0=gamma0, sf=sf,
                              nBatch=nBatch, taskID=taskID)
            # DPParam, newinitname = DPObj.fit(z_batch)
            if isOnlineRead:
                ## load DP model

                dp_model, dp_info_dict, DPParam = DPObj.fitWithPrevModel(z_batch, newinitname, dp_model, dp_info_dict)
                newinitname = dp_info_dict['task_output_path']
            else:
                dp_model, dp_info_dict, DPParam = DPObj.initialFit(z_batch)
                newinitname = dp_info_dict['task_output_path']
        else:
            # if iteration == (num_of_iteration-1) and epoch !=0:
            if iteration != 0 or data_stream_iter != 0:
                # DPParam, newinitname = DPObj.fitWithWarmStart(z_batch, newinitname)
                dp_model, dp_info_dict, DPParam = DPObj.fitWithPrevModel(z_batch, newinitname, dp_model, dp_info_dict)
                newinitname = dp_info_dict['task_output_path']

        trueY = Y[indices]
        fittedY = DPParam['Y']
        ## get the true number of clusters
        trueCluster, counts = np.unique(trueY, return_counts=True)
        trueK = len(trueCluster)
        print(("The true number of cluster is" + " " + str(trueK)))
        print("The proportion of image with true cluster in the batch: \n")
        print(counts / len(trueY))
        clusterResult = clusterEvaluation(trueY, fittedY)
        print("The cluster evaluation result is \n")
        for key, val in clusterResult.items():
            print(key, "=>", val)
        ## get the true cluster and fitted cluster relationship
        dictFitted2True = obtainTrueClusterLabel4AllFittedCluster(trueY, fittedY)
        fittedClusters = dictFitted2True.keys()
        for key in fittedClusters:
            prec = dictFitted2True[key]['prec']
            recall = dictFitted2True[key]['recall']
            trueC = dictFitted2True[key]['trueCluster']
            print("Precision: {}, Recall: {}, fitted: {}, true: {}".format(prec, recall, key, trueC))

        z_fit = sample_output.predict(X, batch_size=batch_size)
        fittedY = obtainFittedYFromDP(DPParam, z_fit)
        accResult = clusterAccuracyUpdated(Y, fittedY)
        # this is the overall accuracy
        acc = accResult['overallRecall']
        print("The current ACC is :{}".format(acc))

        if data_stream_iter == 0 and iteration == 0:
            if flatten:
                vade.compile(optimizer=adam_nn, loss=loss)
            else:
                vade.compile(optimizer=adam_nn, loss=cnn_loss)

        for j in range(batch_iter):
            neg_elbo = vade.train_on_batch(x_batch, x_batch)
            print("Iteration: {}-{}, ELBO: {}".format(iteration, j, -neg_elbo))


# %%
###############################################
## save DP model
dp_model_path = os.path.join(fullOutputPath, 'dp_model.pkl')
dp_model_param = os.path.join(fullOutputPath, 'DPParam.pkl')
dp_info_dict_path = os.path.join(fullOutputPath, 'dp_info_dict.pkl')
joblib.dump(DPParam['model'], dp_model_path)
joblib.dump(DPParam, dp_model_param)
joblib.dump(dp_info_dict['task_output_path'], dp_info_dict_path)
# m : 'm' (# of cluster, latent_dim)
# W : 'B' (# of cluster, latent_dim, latent_dim)
print(outputPath)
m = os.path.join(outputPath, 'm.pkl')
W = os.path.join(outputPath, 'W.pkl')
nu = os.path.join(outputPath, 'nu.pkl')
joblib.dump(DPParam['m'], m)
joblib.dump(DPParam['B'], W)
joblib.dump(DPParam['nu'], nu)

### create images of the posterior component from the trained DP model
### get decoder from VaDE first

encoder, decoder = load_pretrain_vade_weights(encoder, decoder, vade)
generateMeanImage(DPParam, decoder, imgPath='./results/mean_mnist.png')
generateMultipleImgSample(DPParam, decoder, num=10, imgPath='./results/sample_mnist.png')
## save neural network model
# vade.save(os.path.join(outputPath, "vade_DP.hdf5"))
# we should save the model structure and weights seperately.
# serialize model to JSON
# this one is not working for now, don't know how to load self-defined layer
model_json = vade.to_json()
with open(os.path.join(fullOutputPath, "vade_DP_model.json"), "w") as json_file:
    json_file.write(model_json)
# save the weights separately
vade.save_weights(os.path.join(fullOutputPath, "vade_DP_weights.h5"))

if isOnlineWrite:
    online_curr_path = os.path.join(online_path, str(number))
    DPParam_curr_path = os.path.join(online_curr_path, 'DPParam.pkl')
    DPModel_curr_path = os.path.join(online_curr_path, 'dp_model.pkl')
    DPInfo_curr_path = os.path.join(online_curr_path, 'dp_info_dict.pkl')
    joblib.dump(DPParam['model'], DPModel_curr_path)
    joblib.dump(DPParam, DPParam_curr_path)
    joblib.dump(dp_info_dict['task_output_path'], DPInfo_curr_path)
    m = os.path.join(online_curr_path, 'm.pkl')
    W = os.path.join(online_curr_path, 'W.pkl')
    nu = os.path.join(online_curr_path, 'nu.pkl')
    joblib.dump(DPParam['m'], m)
    joblib.dump(DPParam['B'], W)
    joblib.dump(DPParam['nu'], nu)
    generateMeanImage(DPParam, decoder, imgPath=os.path.join(online_curr_path, 'mean_mnist.png'))
    generateMultipleImgSample(DPParam, decoder, num=10, imgPath=os.path.join(online_curr_path, 'sample_mnist.png'))

    with open(os.path.join(online_curr_path, "vade_DP_model.json"), "w") as json_file:
        json_file.write(model_json)
    # save the weights separately
    vade.save_weights(os.path.join(online_curr_path, "vade_DP_weights.h5"))

#########################################################
### use decoder to get all images for all numbers
## generate imagesonline_full_path = os.path.join(online_path, str(number-1))
# for number in range(10):
#    online_curr_path = os.path.join(online_path, str(number))
#    DPParam_path = os.path.join(online_curr_path, 'dp_model.pkl')
#    DPParam = joblib.load(DPParam_path)
#    generateMeanImage(DPParam, decoder, imgPath=os.path.join(online_curr_path, 'total_mean_mnist.png'))









