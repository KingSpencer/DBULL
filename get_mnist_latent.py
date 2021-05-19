import numpy as np

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys

import math
from keras.models import model_from_json
from PIL import Image
import os
import pickle

def load_data():
    path = './dataset/mnist/mnist.pkl.gz'
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
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    X = np.concatenate((x_train,x_test))
    Y = np.concatenate((y_train,y_test))
    return X,Y

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon



batch_size = 100
latent_dim = 10
intermediate_dim = [500,500,2000]
original_dim = 784

def get_model():
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
    sample_output = Model(x, z_mean)

    return vade, sample_output

if __name__ == "__main__":
    vade, sample_output = get_model()
    vade.load_weights('./trained_model_weights/mnist_weights_nn.h5')
    X, Y = load_data()
    latent_z = sample_output.predict(X, batch_size=batch_size)
    latent_mnist = {'z': latent_z, 'y': Y}

    with open('./latent_mnist.pkl', 'wb') as f:
        pickle.dump(latent_mnist, f)
