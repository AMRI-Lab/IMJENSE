#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:28:27 2023

@author: frm
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
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))

#%% load fully sampled k-space data
fpath = './data/knee_slice.h5'
f = h5.File(fpath,'r')
data_cpl = f['KspOrg'][:]

outpath = './output'
if not os.path.exists(outpath):
    os.mkdir(outpath)

Nchl,Nrd,Npe = data_cpl.shape
#%% Parameter settings
Rx = 1   #acceleration rate along the x dimension
Ry = 5  #acceleration rate along the y dimension
num_ACSx = Nrd   #ACS region along the x dimension
num_ACSy = 24 #ACS region along the y dimension
lamda = 3   #TV loss weight. Suggest setting lamda=150 for the macaque data, lamda=5 for the lesion data, lamda=3 for the knee and human brain data.
fn=lambda x: utils.normalize01(np.abs(x))

#%% calculate the sum-of-square ground truth image
img_all = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
gt = np.sqrt(np.sum(np.abs(img_all)**2,0))
 
#%% perform undersampling
tstKsp = data_cpl.transpose(1,2,0)
SamMask = utils.KspaceUnd(Nrd,Npe,Rx,Ry,num_ACSx,num_ACSy)
SamMask = np.tile(np.expand_dims(SamMask,axis=-1),(1,1,Nchl))
tstDsKsp = tstKsp*SamMask

#%% normalize the undersampled k-space      
zf_coil_img=fft.fftshift(fft.ifft2(fft.fftshift(tstDsKsp,axes=(0,1)),axes=(0,1)),axes=(0,1))
NormFactor=np.max(np.sqrt(np.sum(np.abs(zf_coil_img)**2,axis=2)))
tstDsKsp = tstDsKsp/NormFactor
               
time_all_start = time.time()
#pre_img: complex image outputted directly by the MLPs  
#pre_tstCsm: predicted sensitivity maps 
#pre_img_dc: complex image reconstructed after the k-space data consistency step
#pre_ksp: predicted k-space
pre_img, pre_tstCsm, pre_img_dc, pre_ksp = IMJENSE_hash.IMJENSE_hash_Recon(tstDsKsp,SamMask,DEVICE,TV_weight=lamda,PolyOrder=15,MaxIter=200,LrImg = 1e-2,LrCsm=1)
time_all_end = time.time()
print('Reconstruction process costs ',(time_all_end-time_all_start),'secs')        
        
normOrg=fn(gt)
normRec=fn(pre_img_dc)
    
# Note that the psnr and ssim here are computed on the whole image including the background region
# This is different from the results reported in the paper. 
psnrRec=utils.myPSNR(normOrg,normRec)
ssimRec=compute_ssim(normRec,normOrg,data_range=1,gaussian_weights=True)

        
print('{1:.2f} {0:.4f}'.format(psnrRec,ssimRec))
imsave(outpath + '/' + 'gt.png',normOrg)
imsave(outpath + '/' + 'recon.png',normRec)
        

