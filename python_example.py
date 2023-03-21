"""
This code predicts the probe position-integrated scattering cross-sections for a single element fcc crystal. 
The model is trained using the PPISCS class imported from model.py.

Author: Ivan Lobato
Email: Ivanlh20@gmail.com
"""
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
#
# REQUIREMENTS:
# - matplotlib.pyplot
# - numpy
# - PPISCS class imported from model.py

import matplotlib.pyplot as plt
import numpy as np
from model import PPISCS
       
# Input data must be a 2D numpy array with 9 columns
x = np.array([[79, 0, 300, 0.001, -50, 20, 30, 90, 0.085]])

# Load the PPISCS class from model.py
model = PPISCS('coef_scs_fcc.mat')

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