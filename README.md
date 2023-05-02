# IMJENSE
This repository contains the PyTorch implementations of our manuscript "IMJENSE: scan-specific IMplicit representation for Joint coil sENSitivity and image Estimation in parallel MRI"   
IMJENSE was proposed by Ruimin Feng and Dr. Hongjiang Wei. It applies implicit neural representation to the parallel MRI reconstruction.  
## 1. Environmental Requirements  
To run IMJENSE with SIREN network, you should install:  
* Python 3.9.7  
* PyTorch 1.10.2  
* h5py, scikit-image, tqdm  
To run IMJENSE with hash encoding for a faster implementation, you should also install:     
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)  
## 2. Files descriptions
```text
└── IMJENSE/  
      ├── run_demo.py                     # Code to demo how to use IMJENSE for reconstruction  
      ├── IMJENSE.pyc                     # Perform IMJENSE reconstruction
      ├── model_siren.pyc                 # SIREN model
      ├── utils.pyc                       # Some supporting functions
      └── data/  
          ├── knee_slice.h5               # The FastMRI knee data 
          ├── macaque_brain_slice.h5      # The macaque brain data
          ├── human_brain_slice.h5        # The human brain data
          └── lesion_slice.h5             # The lesion data
```
## 3. Usage
You can run "run_demo.py" to test the performance of IMJENSE 
