
import imageio

import math
from tqdm import tqdm
from sklearn.externals import joblib
from numpy.random import multivariate_normal

from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Lambda, Conv2D, Reshape, UpSampling2D, MaxPooling2D, Flatten
from keras.models import Model, load_model
from keras import backend as K

from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import os
import sys
import argparse

from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf
from sklearn.externals import joblib


import sys
# fix path
sys.path.append("../bnpy")
sys.path.append("../bnpy/bnpy")
import numpy as np

import pandas as pd




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



def getDPParam(pklPath):

    with open(pklPath, 'rb') as f:
        DPParam = joblib.load(f)
    return DPParam

def obtainDPParam(DPParam):
    m = DPParam['m']
    W = DPParam['B']
    nu = DPParam['nu']
    beta = DPParam['kappa']
    return m, W, nu, beta

def generateMeanImage(DPParam, decoder, imgPath='./results/mean_mnist.png'):
    # sampling from gaussians
    cluster_sample_list = []
    m, W, nu, beta = obtainDPParam(DPParam)
    z_sample = m
    generated = decoder.predict(z_sample)
    generated = generated.reshape(-1, 28, 28)
    generated *= 255
    generated = generated.astype(np.uint8)
    generated_list = [generated[x] for x in range(generated.shape[0])]
    flattened_generated = np.hstack(generated_list)
    cluster_sample_list.append(flattened_generated)
    merged_sample = np.vstack(cluster_sample_list)
    imageio.imwrite(imgPath, merged_sample)
    return merged_sample


def generateMultipleImgSample(DPParam, decoder, num=10, imgPath='./results/sample_mnist.png'):
    # sampling from gaussians
    cluster_sample_list = []
    m, W, nu, beta = obtainDPParam(DPParam)
    for nc in tqdm(range(len(m))):
        mean = m[nc]
        var = W[nc] * 1 / float(nu[nc])
        z_sample = multivariate_normal(mean, var, num)
        generated = decoder.predict(z_sample)
        generated = generated.reshape(-1, 28, 28)
        # generated = np.minimum(generated * 255 * 1.2, 255)
        generated *= 255
        generated = generated.astype(np.uint8)
        generated_list = [generated[x] for x in range(generated.shape[0])]
        flattened_generated = np.hstack(generated_list)
        cluster_sample_list.append(flattened_generated)
    merged_sample = np.vstack(cluster_sample_list)
    imageio.imwrite(imgPath, merged_sample)
    # imsave(imgPath, merged_sample)
    return merged_sample

def get_models(model_flag, batch_size, original_dim, latent_dim, intermediate_dim):
    if model_flag == "dense":
        x = Input(batch_shape=(batch_size, original_dim))
        h = Dense(intermediate_dim[0], activation='relu')(x)
        h = Dense(intermediate_dim[1], activation='relu')(h)
        h = Dense(intermediate_dim[2], activation='relu')(h)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

        h_decoded = Dense(intermediate_dim[-1], activation='relu')(latent_inputs)
        h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
        h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
        x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

        encoder = Model(x, z, name='encoder')
        decoder = Model(latent_inputs, x_decoded_mean, name='decoder')

        vade = Model(x, decoder(encoder(x)))
        #vade = Model(x, x_decoded_mean)

    elif model_flag.lower() == "cnn":
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
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # constructing several models
        encoder = Model(input_img, z, name='encoder')
        decoder = Model(latent_inputs, decoded, name='decoder')

        decoded_for_vade = decoder(encoder(input_img))
        vade = Model(input_img, decoded_for_vade, name='vade')

        vade.summary()
        encoder.summary()
        decoder.summary()

    return vade, encoder, decoder

def get_temp_vade(batch_size, original_dim, latent_dim, intermediate_dim):
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
    x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    vade = Model(x, x_decoded_mean)
    return vade

def load_pretrain_vade_weights(encoder, decoder, vade_temp):
    encoder.layers[1].set_weights(vade_temp.layers[1].get_weights())
    encoder.layers[2].set_weights(vade_temp.layers[2].get_weights())
    encoder.layers[3].set_weights(vade_temp.layers[3].get_weights())
    encoder.layers[4].set_weights(vade_temp.layers[4].get_weights())
    decoder.layers[-1].set_weights(vade_temp.layers[-1].get_weights())
    decoder.layers[-2].set_weights(vade_temp.layers[-2].get_weights())
    decoder.layers[-3].set_weights(vade_temp.layers[-3].get_weights())
    decoder.layers[-4].set_weights(vade_temp.layers[-4].get_weights())
    return encoder, decoder

def load_pretrain_online_weights(vade, online_path, number, delta=1):
    # OnlineModelFolder = os.path.join(online_path, str(number-1))
    OnlineModelFolder = os.path.join(online_path, str(number-delta))
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


def get_vade(batch_size=128, original_dim=784, intermediate_dim=[500, 500, 2000], latent_dim=10):
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
    x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

    # sample_output = Model(x, z_mean)

    vade = Model(x, x_decoded_mean)
    return vade

def GenerateMultipleImageInOrder(input_path, number, order_file_name='order.txt', delta=0, model_flag='dense', \
                             batch_size=128, original_dim = 784, dim_2d =28, \
                             latent_dim=10, intermediate_dim=[500, 500, 2000], \
                             imgName=None, numOfSamples=6):
    vade_ini, encoder, decoder = get_models(model_flag, batch_size, original_dim, latent_dim,
                                            intermediate_dim)

    vade = get_vade(batch_size, original_dim, intermediate_dim, latent_dim)
    vade = load_pretrain_online_weights(vade, input_path, number, delta)

    encoder, decoder = load_pretrain_vade_weights(encoder, decoder, vade)
    ##### read order.txt file and get the image order #######
    order_file_path = os.path.join(os.path.join(input_path, str(number)), order_file_name)
    ### read txt file in python #######
    order = pd.read_csv(order_file_path, sep=" ", header=0).values

    ##########  read DPParam  ########
    DPParam_path = os.path.join(os.path.join(input_path,str(number)), 'DPParam.pkl')
    DPParam = joblib.load(DPParam_path)

    cluster_sample_list = []
    m, W, nu, beta = obtainDPParam(DPParam)

    for nc in tqdm(range(len(m))):
        mean = m[nc]
        var = W[nc] * 1 / float(nu[nc])
        z_sample = multivariate_normal(mean, var, numOfSamples)
        generated = decoder.predict(z_sample)
        generated = generated.reshape(-1, 28, 28)
        # generated = np.minimum(generated * 255 * 1.2, 255)
        generated *= 255
        generated = generated.astype(np.uint8)
        generated_list = [generated[x] for x in range(generated.shape[0])]
        flattened_generated = np.hstack(generated_list)
        cluster_sample_list.append(flattened_generated)
    ## reorder the images in the list according to order.txt
    nClusters = len(cluster_sample_list)
    ordered_cluster_sample_list = []
    for i in range(nClusters):
        ordered_cluster_sample_list.append(cluster_sample_list[int(order[i, 0])])

    merged_sample = np.vstack(ordered_cluster_sample_list)
    if not imgName is None:
        img_full_name = os.path.join(input_path, imgName)
    else:
        img_full_name = os.path.join(os.path.join(input_path, str(number)), 'ordered_mutiple_img.png')

    imageio.imwrite(img_full_name, merged_sample)
    return merged_sample





def GenerateMeanImageInOrder(input_path, number, order_file_name='order.txt', delta=0, model_flag='dense', \
                             batch_size=128, original_dim = 784, dim_2d =28, \
                             latent_dim=10, \
                             intermediate_dim=[500, 500, 2000], imgName=None):

    vade_ini, encoder, decoder = get_models(model_flag, batch_size, original_dim, latent_dim,
                                            intermediate_dim)

    vade = get_vade(batch_size, original_dim, intermediate_dim, latent_dim)
    vade = load_pretrain_online_weights(vade, input_path, number, delta)

    encoder, decoder = load_pretrain_vade_weights(encoder, decoder, vade)
    ##### read order.txt file and get the image order #######
    order_file_path = os.path.join(os.path.join(input_path, str(number)), order_file_name)
    ### read txt file in python #######
    order = pd.read_csv(order_file_path, sep=" ", header=0).values

    ##########  read DPParam  ########
    DPParam_path = os.path.join(os.path.join(input_path,str(number)), 'DPParam.pkl')
    DPParam = joblib.load(DPParam_path)

    cluster_sample_list = []
    m, W, nu, beta = obtainDPParam(DPParam)
    z_sample = m
    generated = decoder.predict(z_sample)
    generated = generated.reshape(-1, dim_2d, dim_2d)
    generated *= 255
    generated = generated.astype(np.uint8)
    generated_list = [generated[x] for x in range(generated.shape[0])]
    generated_ordered_list = []
    nClusters = len(generated_list)
    for i in range(nClusters):
        generated_ordered_list.append(generated_list[int(order[i, 0])])

    flattened_generated = np.hstack(generated_ordered_list)
    cluster_sample_list.append(flattened_generated)

    merged_sample = np.vstack(cluster_sample_list)
    if not imgName is None:
        img_full_name = os.path.join(input_path, imgName)
    else:
        img_full_name = os.path.join(os.path.join(input_path, str(number)), 'ordered_mean.png')

    imageio.imwrite(img_full_name, merged_sample)
    return merged_sample


if __name__ == "__main__":
    numbers = np.array([1, 3, 5, 7, 9])
    input_path = '/Users/crystal/Documents/VaDE_results/singledigit2'
    for number in numbers:
        GenerateMeanImageInOrder(input_path, number, order_file_name='order.txt', delta=0, model_flag='dense', \
                             batch_size=128, original_dim = 784, dim_2d =28, \
                             latent_dim=10, \
                             intermediate_dim=[500, 500, 2000], imgName=None)

        GenerateMultipleImageInOrder(input_path, number, order_file_name='order.txt', delta=0, model_flag='dense', \
                             batch_size=128, original_dim = 784, dim_2d =28, \
                             latent_dim=10, \
                             intermediate_dim=[500, 500, 2000], imgName=None)

