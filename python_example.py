import matplotlib.pyplot as plt
import ilm_scs_fcc as fcn

fig = plt.figure(1)

ax1 = fig.add_subplot(121)
x = [79, 0, 300, 0.001, -50, 20, 30, 90, 0.085]
y_p = ilm_scs_fcc(x)
ax1.plot(y_p, '-r')
ax1.set_xlabel("Number of atoms")
ax1.set_ylabel("SCS")
ax1.set_aspect(1.25)

ax2 = fig.add_subplot(122)
x = [79, 1, 300, 0.001, -50, 20, 30, 90, 0.085]
y_p = fcn.ilm_scs_fcc(x)
ax2.plot(y_p, '-r')
ax2.set_xlabel("Number of atoms")
ax2.set_ylabel("SCS")
ax2.set_aspect(1.25)

ax2.set_xlim([0, 40])

plt.show()