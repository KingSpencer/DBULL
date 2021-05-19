#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:03:50 2019

@author: crystal
"""

import sys
import argparse
import os

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-bnpyPath', action='store', type = str, dest='bnpyPath', default='/Users/crystal/Documents/bnpy/', \
                    help='path to bnpy code repo')
parser.add_argument('-outputPath', action='store', type = str, dest='outputPath', default='/Users/crystal/Documents/VaDE_results', \
                    help='path to output')
parser.add_argument('-rootPath', action='store', type = str, dest='rootPath', default='/Users/crystal/Documents/VaDE', \
                    help='root path to VaDE')
parser.add_argument('-conv', action='store_true', \
                    help='using convolutional autoencoder or not')
parser.add_argument('-Kmax', action='store', type = int, dest='Kmax',  default=10, help='the maximum number of clusters in DPMM')
parser.add_argument('-dataset', action='store', type = str, dest='dataset',  default = 'reuters10k', help='the options can be mnist,reuters10k and har')
parser.add_argument('-epoch', action='store', type = int, dest='epoch', default = 20, help='The number of epochs')
parser.add_argument('-batch_iter', action='store', type = int, dest='batch_iter', default = 10, help='The number of updates in SGVB')
parser.add_argument('-scale', action='store', type = float, dest='scale', default = 0.005, help='the scale parameter in the loss function')
parser.add_argument('-sf', action='store', type = float, dest='sf', default=0.1, help='the prior diagonal covariance matrix for Normal mixture in DP')
parser.add_argument('-gamma0', action='store', type = float, dest='gamma0', default=5.0, help='hyperparameters for DP in Beta dist')
parser.add_argument('-gamma1', action='store', type= float, dest='gamma1', default=1.0, help='hyperparameters for DP in Beta dist')
parser.add_argument('-logFile', action='store_true', dest='logFile', help='if logfile exists, save the log file to txt')
parser.add_argument('-useLocal', action='store_true', dest='useLocal', help='if use Local, rep environment variable will not be used')
parser.add_argument('-rep', action='store', type=int, dest = 'rep', default=1, help='add replication number as argument')
parser.add_argument('-nLap', action='store', type=int, dest = 'nLap', default=500, help='the number of laps in DP')  
parser.add_argument('-batchsize', action='store', type = int, dest='batchsize', default = 5000, help='the default batch size when training neural network')
parser.add_argument('-threshold', action='store', type=float, dest='threshold', default = 0.88, help= 'stopping criteria')  
parser.add_argument('-nBatch', action='store', type = int, dest='nBatch', default = 5, help='number of batches in DP')
parser.add_argument('-taskID', action='store', type=int, dest = 'taskID', default=1, help='use taskID to random seed for bnpy') 
parser.add_argument('-useNewPretrained', action='store_true',  dest='useNewPretrained', help='Indicator about using new pretrained weights')
parser.add_argument('-useUnsupervised', action='store_true', help='if true, use the original latent representation from the author')
parser.add_argument('-learningRate', action='store', type=float, dest='lr', default=0.01, help='the learning rate in adam_nn')
parser.add_argument('-datasetPath', action='store', type=str, dest='datasetPath', help='the path for new cluster dataset')
parser.add_argument('-initModelPath', action='store', type=str, help='the path for the initial model of DP')


results = parser.parse_args()

# results.useLocal=True
if results.useLocal:
    rep = results.rep
else:
    rep = os.environ["rep"]
    rep = int(float(rep))
    
bnpyPath = results.bnpyPath
outputPath = results.outputPath
rootPath = results.rootPath
sys.path.append(bnpyPath)
subdir = os.path.join(bnpyPath, 'bnpy')
sys.path.append(subdir)
sys.path.append(rootPath)
Kmax = results.Kmax
dataset = results.dataset
threshold = results.threshold
taskID = results.taskID

import bnpy
from data.XData import XData

class DP:
    
    def __init__(self, output_path=outputPath, nLap=300, nTask=1, nBatch=5,sF=0.1, ECovMat='eye',
    K=1, initname='randexamples',moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4, doSaveToDisk=True, gamma1=1.0, gamma0=5.0, Kmax=50, taskID = taskID,  **kwargs):
        self.output_path = output_path
        self.nLap = nLap
        self.nTask = nTask
        self.nBatch = nBatch
        self.sF = sF
        self.ECovMat = ECovMat
        self.gamma1 = gamma1
        self.gamma0 = gamma0
        self.m_startLap = m_startLap
        self.initname=initname
        self.moves= moves
        self.m_startLap=m_startLap
        self.b_startLap = b_startLap
        self.b_Kfresh = b_Kfresh
        self.doSaveToDisk = doSaveToDisk
        self.Kmax = Kmax
        self.K = K
        self.taskID = taskID
    
    
    def run(self, data, mixModel='DPMixtureModel', obsModel='Gauss', alg='memoVB'):
        dp_model, dp_info_dict =bnpy.run(data, mixModel, obsModel, alg, K = self.K, output_path=self.output_path,
                                        nLap = self.nLap, nTask=self.nTask, nBatch=self.nBatch, sF=self.sF,
                                        ECovMat=self.ECovMat, m_startLap=self.m_startLap, initname=self.initname,
                                        moves=self.moves, b_startLap=self.b_startLap, b_Kfresh=self.b_Kfresh, 
                                        doSaveToDisk=self.doSaveToDisk, gamma1=self.gamma1, gamma0 = self.gamma0,
                                        Kmax = self.Kmax, taskID = self.taskID)
        return dp_model, dp_info_dict


    def initialFit(self, z_batch):
        if isinstance(z_batch, XData):
            data = z_batch
        else:
            data = XData(z_batch, dtype='auto')
        dp_model, dp_info_dict = self.run(data)
        DPParam = self.extractDPParam(dp_model, data)
        return dp_model, dp_info_dict, DPParam


    def fit(self, z_batch):
        if isinstance(z_batch, XData):
            data = z_batch
        else:
            data = XData(z_batch, dtype='auto')
        dp_model, dp_info_dict = self.run(data)
        DPParam = self.extractDPParam(dp_model, data)
        return DPParam, dp_info_dict['task_output_path']

    def fitWithPrevModel(self, z_batch, initname, dp_model, dict_info):
        if isinstance(z_batch, XData):
            data = z_batch
        else:
            data = XData(z_batch, dtype='auto')
        dp_model, dp_info_dict = bnpy.runWithInit(dataName=data, allocModelName='DPMixtureModel', obsModelName='Gauss', algName='memoVB',
                                                  K=self.K, output_path=self.output_path,
                                                  nLap=self.nLap, nTask=self.nTask, nBatch=self.nBatch, sF=self.sF,
                                                  ECovMat=self.ECovMat, m_startLap=self.m_startLap,
                                                  initname=initname,
                                                  moves=self.moves, b_startLap=self.b_startLap, b_Kfresh=self.b_Kfresh,
                                                  doSaveToDisk=self.doSaveToDisk, gamma1=self.gamma1,
                                                  gamma0=self.gamma0,
                                                  Kmax=self.Kmax, taskID=self.taskID,
                                                  hmodel=dp_model, dict_info=dict_info)

        DPParam = self.extractDPParam(dp_model, data)
        # DPParam, dp_info_dict['task_output_path']
        return dp_model, dp_info_dict, DPParam



    
    def fitWithWarmStart(self, z_batch, initname):
        if isinstance(z_batch, XData):
            data = z_batch
        else:
            data = XData(z_batch, dtype='auto')
        self.initname = initname    
        dp_model, dp_info_dict = self.run(data)

        DPParam = self.extractDPParam(dp_model, data)
        return DPParam, dp_info_dict['task_output_path']
            
    

    def extractDPParam(self, model, dataset):
        LP = model.calc_local_params(dataset)
        LPMtx = LP['E_log_soft_ev']
        ## to obtain hard assignment of clusters for each observation
        Y = LPMtx.argmax(axis=1)
    
        ## obtain sufficient statistics from DP
        SS = model.get_global_suff_stats(dataset, LP, doPrecompEntropy=1)
        Nvec = SS.N
    
        ## get the number of clusters
        K = model.obsModel.Post.K
    
        m = model.obsModel.Post.m
    
        # get the posterior covariance matrix
        B = model.obsModel.Post.B
    
        # degree of freedom
        nu = model.obsModel.Post.nu
    
        # scale precision on parameter m, which is lambda parameter in wiki for Normal-Wishart dist
        kappa = model.obsModel.Post.kappa
    
        ## get the ELBO of the DP given z_batch
        elbo = model.calc_evidence(dataset, SS, LP)
        
        
        ## save the variables in a dictionary
        DPParam = dict()
        DPParam['LPMtx'] = LPMtx
        DPParam['Y'] = Y
        DPParam['Nvec'] = Nvec
        DPParam['K'] = K
        DPParam['m'] = m
        DPParam['B'] = B
        DPParam['nu'] = nu
        DPParam['kappa'] = kappa
        DPParam['elbo'] = elbo
        DPParam['model'] = model
        
        return DPParam

    
        
        
#########################################################
## test and example use of the clas
## correctness of the algorithm has been validated        
#########################################################
## dataset_path = os.path.join(bnpy.DATASET_PATH, 'AsteriskK8')
## dataset = bnpy.data.XData.read_npz(
##    os.path.join(dataset_path, 'x_dataset.npz'))   
## DPObj = DP()   
## dp_model,  dp_info_dict = DPObj.run(dataset)
## DPParam0 = DPObj.extractDPParam(dp_model, dataset)
## DPParam1 = DPObj.fit(dataset)
########################################################

        
        
        
        
        
        
