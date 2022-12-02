% Copyright 2022 Ivan Lobato <Ivanlh20@gmail.com>

clear; clc;
% zone_axis = 0: 110/101/011, 1: 001/100/010
% [Z, zone_axis, E_0, c_30, c_10, cond_lens_outer_aper_ang, det_inner_ang, det_outer_ang, rmsd_3d]


figure(1); clf;
subplot(1, 2, 1);
x = [79, 0, 300, 0.001, -50, 20, 30, 90, 0.085].';
y_p = ilm_scs_fcc(x);
plot(y_p, '-r');
xlabel("Number of atoms")
ylabel("SCS")
pbaspect([1.25 1 1])

subplot(1, 2, 2);
x = [79, 1, 300, 0.001, -50, 20, 30, 90, 0.085].';
y_p = ilm_scs_fcc(x);
plot(y_p, '-r');
xlabel("Number of atoms")
ylabel("SCS")
pbaspect([1.25 1 1])

% xlim([0 40])