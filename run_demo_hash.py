#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:28:27 2023
@author: frm

This script is a demo for hash encoding acceleration

"""

import numpy as np
import h5py as h5
import os
from numpy import fft
import torch
from skimage.metrics import structural_similarity as compute_ssim
from skimage.io import imsave
import utils
import IMJENSE_hash
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))

#%% load fully sampled k-space data
fpath = './data/human_brain_slice.h5'
f = h5.File(fpath,'r')
data_cpl = f['KspOrg'][:]

outpath = './results_hash'
if not os.path.exists(outpath):
    os.mkdir(outpath)

Nchl,Nrd,Npe = data_cpl.shape
#%% Parameter settings
Rx = 1   #acceleration rate along the x dimension
Ry = 4 #acceleration rate along the y dimension
num_ACSx = Nrd   #ACS region along the x dimension
num_ACSy = 24 #ACS region along the y dimension
lamda = 3   #TV loss weight. Suggest setting lamda=250 for the macaque data, lamda=15 for the lesion data, lamda=8 for the knee data, lamda=3 for the human brain data.
encoding_config={
		"otype": "Grid",
        "type": "Hash",
		"n_levels":16,
		"n_features_per_level": 2,   #default 2     
		"log2_hashmap_size": 18,  # default 19
		"base_resolution": 16,    #default 16
		"per_level_scale": 2.0,    #default 2
        "interpolation":"Linear"
        }
network_config={
	"otype": "FullyFusedMLP",    # Component typeSamMask.
	"activation": "ReLU",        # Activation of hidden layers.
	"output_activation": "None", # Activation of the output layer.
	"n_neurons":64,            # Neurons in each hidden layer.
	                             # May only be 16, 32, 64, or 128.
	"n_hidden_layers":1,        # Number of hidden layers.                        
}
# hash encoding configuration for sensitivity maps
encoding_config_csm={
		"otype": "Grid",
        "type0": "Hash",
		"n_levels":8,
		"n_features_per_level": 2,   #default 2     
		"log2_hashmap_size": 18,  # default 19
		"base_resolution": 1,    #default 16
		"per_level_scale": 1.5,    #default 2
        "interpolation":"Linear"
        }

fn=lambda x: utils.normalize01(np.abs(x))

#%% calculate the root of sum-of-square ground truth image
img_all = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
gt = np.sqrt(np.sum(np.abs(img_all)**2,0))
 
#%% perform undersampling
tstKsp = data_cpl.transpose(1,2,0)
SamMask = utils.KspaceUnd(Nrd,Npe,Rx,Ry,num_ACSx,num_ACSy)
SamMask = np.tile(np.expand_dims(SamMask,axis=-1),(1,1,Nchl))
tstDsKsp = tstKsp*SamMask

#%% Crop the background region along the readout direction to further accelerate the reconstruction (only for the knee data)
# tstDsKsp = fft.fftshift(fft.ifft(fft.fftshift(tstDsKsp,axes=0),axis=0),axes=0)
# tstDsKsp = tstDsKsp[80:560,:,:]
# tstDsKsp = fft.fftshift(fft.fft(fft.fftshift(tstDsKsp,axes=0),axis=0),axes=0)
# SamMask = SamMask[80:560,:,:]
#%% normalize the undersampled k-space      
zf_coil_img=fft.fftshift(fft.ifft2(fft.fftshift(tstDsKsp,axes=(0,1)),axes=(0,1)),axes=(0,1))
NormFactor=np.max(np.sqrt(np.sum(np.abs(zf_coil_img)**2,axis=2)))
tstDsKsp = tstDsKsp/NormFactor

#pre_img: complex image outputted directly by the MLPs  
#pre_tstCsm: predicted sensitivity maps 
#pre_img_dc: complex image reconstructed after the k-space data consistency step
#pre_ksp: predicted k-space
pre_img, pre_tstCsm, pre_img_dc, pre_img_sos, pre_ksp = IMJENSE_hash.IMJENSE_hash_Recon(tstDsKsp,SamMask,DEVICE,encoding_config,network_config,encoding_config_csm,TV_weight=lamda,MaxIter=200,LrImg = 1e-2,LrCsm=1e-2)


#%%
normOrg=fn(gt)
normRec=fn(pre_img_dc)
    
# Note that the psnr and ssim here are computed on the whole image including the background region
# This is different from the results reported in the paper. 
psnrRec=utils.myPSNR(normOrg,normRec)
ssimRec=compute_ssim(normRec,normOrg,data_range=1,gaussian_weights=True)

        
print('{1:.2f} {0:.4f}'.format(psnrRec,ssimRec))
imsave(outpath + '/' + 'gt.png',normOrg)
imsave(outpath + '/' + 'recon.png',normRec)
        

