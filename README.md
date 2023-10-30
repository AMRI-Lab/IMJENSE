# IMJENSE
This repository contains the PyTorch implementations of our manuscript "IMJENSE: scan-specific IMplicit representation for Joint coil sENSitivity and image Estimation in parallel MRI"   
IMJENSE was proposed by Ruimin Feng and Dr. Hongjiang Wei. It applies implicit neural representation to the parallel MRI reconstruction.  
## 1. Environmental Requirements  
### To run "run_demo.py" for reconstruction using IMJENSE with SIREN network, you should install:  
* Python 3.9.7  
* PyTorch 1.10.2  
* h5py, scikit-image, tqdm  
### To run "run_demo_hash.py" for reconstruction using IMJENSE with hash encoding for a faster implementation, you should also install:     
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
### To run "parameter_tuning.py" for hyperparameter tuning using the Bayesian optimization method, you should also install:     
* [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)
## 2. Files descriptions
```text
└── IMJENSE/  
      ├── run_demo.py                     # Code to demo how to use IMJENSE for reconstruction
      ├── run_demo_hash.py                # Code to demo how to use IMJENSE with hash encoding for a faster implementation
      ├── parameter_tuning.py             # Code to demo how to use the Bayesian optimization method for hyperparameter tuning
      ├── IMJENSE.pyc                     # Perform IMJENSE reconstruction
      ├── IMJENSE_hash.pyc                # Perform IMJENSE reconstruction with the hash encoding
      ├── model_siren.pyc                 # SIREN model
      ├── utils.pyc                       # Some supporting functions
      └── data/  
          ├── knee_slice.h5               # The FastMRI knee data 
          ├── macaque_brain_slice.h5      # The macaque brain data
          ├── human_brain_slice.h5        # The human brain data
          └── lesion_slice.h5             # The lesion data
```
## 3. Usage
You can run "run_demo.py" to test the performance of IMJENSE and run "run_demo_hash.py" to test the performance of IMJENSE with hash encoding 
