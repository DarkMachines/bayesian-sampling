# Bayesian-sampling
Framework for the Bayesian sampling challenge of DarkMachines.org. 

# Run

The script **hds_bilby_ml.py** implements four analytical functions: Rosenbrock, Rastrigin, Himmelblau, EggBox and GaussianShells  
plus the LambdaCDM and the MSSM7 log likelihoods in form of Neural Nets (Details about the implementation of each can be 
found in the script). It runs in [bilby](https://github.com/lscsoft/bilby) framework.

Currently it runs dynesty in its dynamic nested sampling realization and plots 1D and 2D posteriors in a corner plot. 
