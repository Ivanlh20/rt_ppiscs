# Real-time simulations of ADF STEM probe position-integrated scattering cross-sections for single element fcc crystals in zone axis orientation using a densely connected neural network

## Overview
Quantifying annular dark field (ADF) scanning transmission electron microscopy (STEM) images in terms of composition or thickness often relies on probe-position integrated scattering cross sections (PPISCSs). In order to compare experimental PPISCSs with theoretically predicted ones, expensive simulations are needed for a given specimen, zone axis orientation, and a variety of microscope settings. The computation time of such simulations can be lengthy, even when using a single GPU.

To address this issue, we present rt_ppiscs, a densely connected neural network that is able to perform real-time ADF STEM PPISCS predictions as a function of atomic column thickness for the most common fcc crystals along their main zone axis orientations, root-mean-square displacements, and microscope parameters. Our proposed architecture is parameter efficient and yields accurate predictions for the PPISCS values for a wide range of input parameters commonly used in aberration-corrected transmission electron microscopes.

This repository contains the code for rt_ppiscs, a tool for real-time prediction of atomic column thickness using annular dark field scanning transmission electron microscopy (STEM). It includes the inference code, the source code for training, and the network architecture. rt_ppiscs was developed by Ivan Lobato (Ivanlh20@gmail.com).

## Requirements
Currently, rt_ppiscs supports the following three platforms:
- MATLAB 2017+
- Python 3.8+
- Tensorflow 2.10+

## Performance
The architecture of rt_ppiscs was optimized to run on a normal desktop computer, so it does not require the use of GPU acceleration.

## Usage
Refer to the examples files for detailed instructions on how to use rt_ppiscs.

**Please cite rt_ppiscs in your publications if it helps your research:**
```bibtex
    @article{LBV_2023,
      Author = {I.Lobato and A. De Backer and S.Van Aert},
      Journal = {Ultramicroscopy},
      Title = {Real-time simulations of ADF STEM probe position-integrated scattering cross-sections for single element fcc crystals in zone axis orientation using a densely connected neural network},
      Year = {2023},
  	  volume  = {xxx},
      pages   = {xxx-xxx}
    }