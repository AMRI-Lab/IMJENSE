#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:58:52 2023
@author: frm

This script shows how to use the Bayesian optimization method to 
fine-tune the hyperparameters lambda and w0 

"""
import numpy as np
import h5py as h5
import torch
import os
from numpy import fft
import utils
import IMJENSE
from skimage.io import imsave
from scipy.io import loadmat
from scipy.io import savemat
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))
fn=lambda x: utils.normalize01(np.abs(x))

outpath = './parameter_tune_BayesSearch'
if not os.path.exists(outpath):
    os.mkdir(outpath)

#%% load the training data from one subject
rpath = '/mnt/288T/frm/IMJENSE_datasets/FastMRI_knee/testing'
fpath = rpath + '/' + 'file1001759.h5'   
f = h5.File(fpath,'r')
data_cpl = f['kspace'][:]
Nsli,Nchl,Nrd,Npe = data_cpl.shape
    
img_all = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
gt = np.sqrt(np.sum(np.abs(img_all)**2,1))

tmask = loadmat(rpath+'/'+'tmask_file1001759.mat')['tmask'].transpose(2,1,0)
    
num_ACSx = Nrd
num_ACSy = 24
Rx = 1
Ry = 4

#%% define the black-box target function
def target_function(w0,Lambda):   
    w0 = np.round(w0)
               
    recon_img = np.zeros((Nsli,Nrd,Npe),dtype = np.complex64)
    recon_csm = np.zeros((Nsli,Nchl,Nrd,Npe),dtype = np.complex64)
    recon_img_dc = np.zeros((Nsli,Nrd,Npe),dtype = np.complex64)
    recon_ksp = np.zeros((Nsli,Nchl,Nrd,Npe),dtype = np.complex64)
    evalmat = np.zeros((Nsli))
    
    for sli in range(Nsli):
        tstKsp = data_cpl[sli].transpose(1,2,0)
    
        SamMask = utils.KspaceUnd(Nrd,Npe,Rx,Ry,num_ACSx,num_ACSy)
        SamMask = np.tile(np.expand_dims(SamMask,axis=-1),(1,1,Nchl))
        tstDsKsp = tstKsp*SamMask
    
        zf_coil_img=fft.fftshift(fft.ifft2(fft.fftshift(tstDsKsp,axes=(0,1)),axes=(0,1)),axes=(0,1))
        NormFactor=np.max(np.sqrt(np.sum(np.abs(zf_coil_img)**2,axis=2)))
        tstDsKsp = tstDsKsp/NormFactor
    
        pre_img, pre_tstCsm, pre_img_dc, pre_img_sos, pre_ksp  = IMJENSE.IMJENSE_Recon(tstDsKsp,SamMask,DEVICE,w0=w0,TV_weight=Lambda,PolyOrder=15,MaxIter=1500,LrImg = 1e-4,LrCsm=0.1)
        
        recon_img[sli] = pre_img
        recon_csm[sli] = pre_tstCsm.transpose(2,0,1)
        recon_img_dc[sli] = pre_img_dc
        recon_ksp[sli] = pre_ksp.transpose(2,0,1)

        normOrg=fn(gt[sli])*tmask[sli]
        normRec=fn(pre_img_dc)*tmask[sli]
    
        psnrRec=utils.myPSNR(normOrg,normRec)
        evalmat[sli] = psnrRec
            
    imsave(outpath + '/' + 'gt.png',fn(gt[4]*tmask[4]))
    imsave(outpath + '/' + 'recon_w0'+str(w0)+'_TV'+str(Lambda)+'.png',fn(recon_img_dc[4]*tmask[4]))
    savemat(outpath+'/'+'recon_w0'+str(w0)+'_TV'+str(Lambda)+'.mat',
        {'recon_img':recon_img,
         'recon_csm':recon_csm,
         'recon_img_dc':recon_img_dc,
         'recon_ksp':recon_ksp,
         'evalmat':evalmat})
    return np.mean(evalmat) 
    
#%% Bounded region of parameter space
pbounds = {'w0':(10,50),'Lambda':(0,100)}
logger = JSONLogger(path=outpath+'/'+'logs')
optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds, 
    verbose=2,
    random_state=1,
    allow_duplicate_points=True
)
optimizer.subscribe(Events.OPTIMIZATION_STEP,logger)
optimizer.maximize(init_points=4, n_iter=20)


print("Optimal value of w0:", optimizer.max['params']['w0'])
print("Optimal value of Lambda:", optimizer.max['params']['Lambda'])
print("Optimal function value:", optimizer.max['target'])

