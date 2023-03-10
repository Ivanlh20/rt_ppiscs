# Copyright 2023 Ivan Lobato <Ivanlh20@gmail.com>
# zone_axis = 0: 110/101/011, 1: 001/100/010
# [Z, zone_axis, E_0, c_30, c_10, cond_lens_outer_aper_ang, det_inner_ang, det_outer_ang, rmsd_3d]

import matplotlib.pyplot as plt
import numpy as np
from ilp_scs_fcc import ilp_scs_fcc

x = np.array([79, 0, 300, 0.001, -50, 20, 30, 90, 0.085]).reshape((-1, 1))
y_p = ilp_scs_fcc(x)

# plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
# plt.text(0.5, 0.5, "Some Text", horizontalalignment='center',
#      verticalalignment='center', transform=plt.gca().transAxes)
# plt.show()

plt.figure(1)
plt.plot(y_p, '-r')
plt.xlabel('Number of atoms', fontsize = 14)
plt.ylabel('Scatering cross-sections (Å^2)', fontsize = 14)
str_text_p = ['Z = {:d}'.format(int(x.take(0))),
            'Zone axis = {}'.format('001' if x.take(1)>0.5 else '110'),
            'E_0 = {:d}keV'.format(int(x.take(2))),
            'Cs = {:3.1f}um'.format(1000*x.take(3)),
            'Def = {:4.1f}Å'.format(x.take(4)),
            'A. Rad = {:4.1f}mrad'.format(x.take(5)),
            'Inner = {:4.1f}mrad'.format(x.take(6)),
            'Outer = {:4.1f}mrad'.format(x.take(7)),
            'Rmsd = {:4.3f}Å'.format(x.take(8))]

xp = np.ones((9,))*0.55
yp = 0.05+np.linspace(0.55, 0.0, 9)
for x_t, y_t, str_p in zip(xp, yp, str_text_p):
	plt.text(x_t, y_t, str_p, fontsize = 13, transform=plt.gca().transAxes)
 
plt.show()