% COPYRIGHT 2023 IVAN LOBATO <IVANLH20@GMAIL.COM>
% 
% DESCRIPTION:
% This code uses the PPISCS_Model class imported from PPISCS_Model.py to predict the probe position-integrated scattering cross-sections for a single element fcc crystal.
%
% INPUT:
% The input data must be a 2D numpy array with 9 columns:
%     - Z: atomic number
%     - zone_axis: zone axis, which can take the values 0 for zone orientations 110/101/011 and 1 for zone axis orientation 001/100/010
%     - E_0: incident electron energy
%     - c_30: spherical aberration
%     - c_10: defocus
%     - cond_lens_outer_aper_ang: condenser lens aperture semi-angle
%     - det_inner_ang: detector inner angle
%     - det_outer_ang: detector outer angle
%     - rmsd_3d: root mean square displacement 
%
% REQUIREMENTS:
% - PPISCS_Model class

clear; clc;
figure(1); clf;

% Input data must be a 2D numpy array with 9 columns
x = [79, 0, 300, 0.001, -50, 20, 30, 90, 0.085];
model = PPISCS_Model('coef_scs_fcc.mat');
y_p = model.predict(x);

% Plot the predictions
figure(1); clf;
plot(y_p, '-r', 'LineWidth', 1.5);
xlabel('Number of atoms', 'fontname', 'times', 'fontsize', 14);
ylabel('Scattering cross-sections (Å^2)', 'fontname', 'times', 'fontsize', 14);
pbaspect([1.25 1, 1]);

% Add text to the plot describing the input data
str_text{1} = ['Z = ', num2str(x(1), '%d')];
str_text{2} = ['Zone axis = ', ilm_ifelse(x(2)==1, '001', '110')];
str_text{3} = ['E_0 = ', num2str(x(3), '%d'), ' keV'];
str_text{4} = ['Cs = ', num2str(1000*x(4), '%3.1f'), ' μm'];
str_text{5} = ['Def = ', num2str(x(5), '%4.1f'), ' Å'];
str_text{6} = ['A. Rad = ', num2str(x(6), '%4.1f'), ' mrad'];
str_text{7} = ['Inner = ', num2str(x(7), '%4.1f'), ' mrad'];
str_text{8} = ['Outer = ', num2str(x(8), '%4.1f'), ' mrad'];
str_text{9} = ['Rmsd = ', num2str(x(9), '%4.3f'), ' Å'];
xt = repmat(0.55, 1, 9);
yt = 0.05 + linspace(0.55, 0.0, 9);
text(xt, yt, str_text, 'units','normalized', 'fontname', 'times', 'fontsize', 13);
