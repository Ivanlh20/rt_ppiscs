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

## Pip installation
To install **rt_ppiscs** and its dependencies, run the following command in your terminal or command prompt:
```
pip install rt_ppiscs
```

**Python example**
```python
# INPUT:
# The input data must be a 2D numpy array with 9 columns:
#     - Z: atomic number
#     - zone_axis: zone axis, which can take the values 0 for zone orientations 110/101/011 and 1 for zone axis orientation 001/100/010
#     - E_0: incident electron energy
#     - c_30: spherical aberration
#     - c_10: defocus
#     - cond_lens_outer_aper_ang: condenser lens aperture semi-angle
#     - det_inner_ang: detector inner angle
#     - det_outer_ang: detector outer angle
#     - rmsd_3d: root mean square displacement

import matplotlib.pyplot as plt
import numpy as np
from rt_ppiscs.model import PPISCS

# Input data must be a 2D numpy array with 9 columns
x = np.array([[79, 0, 300, 0.001, -50, 20, 30, 90, 0.085]])

# Load the PPISCS_Model class from PPISCS_Model.py
model = PPISCS()

# Make predictions using the PPISCS_Model class
y_p = model.predict(x)

# Plot the predictions
plt.figure(1)
plt.plot(y_p.T, '-r')
plt.xlabel('Number of atoms', fontsize=14)
plt.ylabel('Scattering cross-sections (Å^2)', fontsize=14)

# Add text to the plot describing the input data
str_text_p = ['Z = {:d}'.format(int(x.take(0))),
              'Zone axis = {}'.format('001' if x.take(1) > 0.5 else '110'),
              'E_0 = {:d}keV'.format(int(x.take(2))),
              'Cs = {:3.1f}um'.format(1000 * x.take(3)),
              'Def = {:4.1f}Å'.format(x.take(4)),
              'A. Rad = {:4.1f}mrad'.format(x.take(5)),
              'Inner = {:4.1f}mrad'.format(x.take(6)),
              'Outer = {:4.1f}mrad'.format(x.take(7)),
              'Rmsd = {:4.3f}Å'.format(x.take(8))]

# Add the text to the plot
xp = np.ones((9,)) * 0.55
yp = 0.05 + np.linspace(0.55, 0.0, 9)
for x_t, y_t, str_p in zip(xp, yp, str_text_p):
    plt.text(x_t, y_t, str_p, fontsize=13, transform=plt.gca().transAxes)

plt.show()
```

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