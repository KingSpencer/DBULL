import numpy as np
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

from scipy.misc import imsave

import math
from tqdm import tqdm
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
import tensorflow as tf
from sklearn.externals import joblib

# fix path
sys.path.append("../bnpy")
sys.path.append("../bnpy/bnpy")

class DPVAE_Generator:
    def __init__():
        pass

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

'''def load_pretrain_vade(vade, root_path="/home/zifeng/Research/VaDE/results"):
    batch_size, latent_dim = 128, 10
    path = os.path.join(root_path, 'vade_DP.hdf5')
    prev_vade = load_model(path)
    for i in range(1, 5):
        vade.layers[i].set_weights(prev_vade.layers[i].get_weights())
        vade.layers[-i].set_weights(prev_vade.layers[-i].get_weights())
    return vade'''

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

if __name__ == "__main__":
    batch_size = 128
    original_dim = 784
    latent_dim = 10
    intermediate_dim = [500,500,2000]
    model_flag = 'dense'
    vade, encoder, decoder = get_models(model_flag, batch_size, original_dim, latent_dim, intermediate_dim)
    if model_flag == 'dense':
        vade_temp = get_temp_vade(batch_size, original_dim, latent_dim, intermediate_dim)
        #vade.summary()
        #encoder.summary()
        #decoder.summary()
        # encoder, decoder = load_pretrain_weights(encoder, decoder)
        vade_temp.load_weights("./best_mnist_results/vade_DP_weights.h5")
        #vade_temp.load_weights("/home/zifeng/Research/DPVAE/results/vade_DP_weights.h5")
        print("************* weights loaded successfully! **************")
        encoder, decoder = load_pretrain_vade_weights(encoder, decoder, vade_temp)
    elif model_flag == 'cnn':
        vade.load_weights("./best_cnn_results/vade_DP_weights.h5")
    # test loading and saving model structure
    #jj = vade.to_json()
    #aa = model_from_json(jj)
    # not working ...
    # print(len(vade.get_weights()))
    # TODO: Load DP parameters and generate new data
    # with open('./results/m.pkl', 'rb') as f:
    #    m = joblib.load(f)
    # with open('./results/W.pkl', 'rb') as f:
    #    W = joblib.load(f)
    if model_flag == 'dense':

        with open('./best_mnist_results/DPParam.pkl', 'rb') as f:
        #with open('./results/DPParam.pkl', 'rb') as f:
            DPParam = joblib.load(f)
    elif model_flag == 'cnn':
        with open('./best_cnn_results/DPParam.pkl', 'rb') as f:
            DPParam = joblib.load(f)
        with open('./results/DPParam.pkl', 'rb') as f:
            DPParam_dense = joblib.load(f)
    m = DPParam['m']
    W = DPParam['B']
    nu = DPParam['nu']
    beta = DPParam['kappa']

    '''if model_flag == 'dense':
        l = [0, 1, 2, 4,5,6,7,8,9,10]
        m = m[l]
        W = W[l]
        nu = nu[l]
        beta = beta[l]'''
    # sampling from gaussians
    cluster_sample_list = []
    print("************* generating new data! **************")
    for nc in tqdm(range(len(m))):
        mean = m[nc]
        #lam = np.linalg.inv(W[nc]) * nu[nc]
        #var = np.linalg.inv(lam)
        var = W[nc] * 1 / float(nu[nc])
        z_sample = multivariate_normal(mean, var, 10)
        # we then feed z_sample to the decoder
        generated = decoder.predict(z_sample)
        generated = generated.reshape(-1, 28, 28)
        #generated = np.minimum(generated * 255 * 1.2, 255)
        generated *= 255
        generated = generated.astype(np.uint8)
        generated_list = [generated[x] for x in range(generated.shape[0])]
        flattened_generated = np.hstack(generated_list)
        cluster_sample_list.append(flattened_generated)
        # print(flattened_generated.shape)
        # print(z_sample.shape)
    merged_sample = np.vstack(cluster_sample_list)
    imsave('./results/sample_best_mnist.png', merged_sample)

    cluster_mean_list = []
    print("************* generating new data with mean! **************")
    for nc in tqdm(range(len(m))):
        mean = m[nc]

        z_sample = np.expand_dims(mean, 0)
        # we then feed z_sample to the decoder
        generated = decoder.predict(z_sample)
        generated = generated.reshape(28, 28)
        #generated = np.minimum(generated * 255 * 1.2, 255)
        generated *= 255
        generated = generated.astype(np.uint8)
        cluster_mean_list.append(generated)
        # print(flattened_generated.shape)
        # print(z_sample.shape)
    merged_mean_sample = np.hstack(cluster_mean_list)
    imsave('./results/mean_sample_best_mnist.png', merged_mean_sample)
