# Copyright 2022 Ivan Lobato <Ivanlh20@gmail.com>
# zone_axis = 0: 110/101/011, 1: 001/100/010
# [Z, zone_axis, E_0, c_30, c_10, cond_lens_outer_aper_ang, det_inner_ang, det_outer_ang, rmsd_3d]

import matplotlib.pyplot as plt
import numpy as np
from model import ilp_scs_fcc

x = np.array([79, 0, 300, 0.001, -50, 20, 30, 90, 0.085]).reshape((-1, 1))
y_p = ilp_scs_fcc(x)

plt.figure(1)
plt.plot(y_p, '-r')
plt.xlabel('Number of atoms', fontname = 'times', fontsize = 14)
plt.ylabel('Scatering cross-sections (Å^2)', fontname = 'times', fontsize = 14)
plt.pbaspect([1.25, 1, 1])

# str_text = [f'Z = {int(x[0])}',
#             f'Zone axis = '001' if x[1] else '110',
#             f'E_0 = {int(x[2])}keV',
#             f'Cs = {1000*x[3]:3.1f}um',
#             f'Def = {x[4]:4.1f}Å',
#             f'A. Rad = {x[5]:4.1f}mrad',
#             f'Inner = {x[6]:4.1f}mrad',
#             f'Outer = {x[7]:4.1f}mrad',
#             f'Rmsd = {x[8]:4.3f}Å']

# xt = np.ones(9)*0.75
# yt = np.linspace(0.6, 0.0, 9)
# plt.text(xt, yt, str_text, fontname = 'times', fontsize = 13)