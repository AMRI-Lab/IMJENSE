# IMJENSE
This repository contains the PyTorch implementations of our manuscript "IMJENSE: scan-specific IMplicit representation for Joint coil sENSitivity and image Estimation in parallel MRI"   
IMJENSE was proposed by Ruimin Feng and Dr. Hongjiang Wei. It applies implicit neural representation to the parallel MRI reconstruction.  
## 1. Environmental Requirements
* Python 3.9.7  
* PyTorch 1.10.2  
* h5py, scikit-image, tqdm  
## 2. Files descriptions
 data  
  - knee_slice.h5: The FastMRI knee data used in the paper  
data/macaque_brain_slice: The macaque brain data used in the paper  
data/human_brain_slice.h5: The human brain data used in the paper  
data/lesion_slice.h5: The lesion data used in the paper

