% Copyright 2022 Ivan Lobato <Ivanlh20@gmail.com>

function[y]=ilm_scs_fcc(x, crop, model)
    % [Z, zone_axis, E_0, c_30, c_10, cond_lens_outer_aper_ang, det_inner_ang, det_outer_ang, rmsd_3d]
    sz = size(x);
    if ((sz(1)~=9) && (sz(2)~=9))
        disp('Wrong input dimensions');
        return;
    end
    
    if nargin==1
        crop = true;
    end
    
    if nargin<3
        model = importdata('coef_scs_fcc.mat', 'model');
    end

    y = fcn_scs_fcc(x, crop, model);
end

function[y]=fcn_scs_fcc(x, crop, model)
    sz = size(x);
    if sz(2)~=9
        x = x.';
    end

    x = double(x);
    y = (x - model.x_sft)./model.x_sc;
    
    y = fcn_eval_bz(y, model.bias, model.weights);
    
    if crop && (x(1, 2)==1)
       y = y(:, 1:61); 
    end

    y = y.';
end

function [y] = fcn_eval_bz(y_i, bias, weights, bz)
    if nargin <4
        bz = 4096*4;
    end
    
    n_y = size(y_i, 1);
    if n_y<=bz
       y = fcn_eval(y_i, bias, weights);
       return
    end
    
    nk = floor(n_y/bz);
    y = zeros(n_y, length(bias{end}));
    
    for ik=1:nk
        ik_0 = (ik-1)*bz + 1;
        ik_e = ik_0 + bz - 1;
        y(ik_0:ik_e, :) = fcn_eval(y_i(ik_0:ik_e, :), bias, weights);
    end

    if n_y>nk*bz
        ik_0 = nk*bz + 1;
        ik_e = n_y;
        y(ik_0:ik_e, :) = fcn_eval(y_i(ik_0:ik_e, :), bias, weights);
    end
end

function [y] = fcn_eval(y, bias, weights)
    n_lay = length(bias);
    
    y = swish(y*weights{1} + bias{1});
        
    for ik=2:(n_lay-1)
        y = [y, swish(y*weights{ik} + bias{ik})];
    end

    y = y*weights{n_lay} + bias{n_lay};
    y = log(1+exp(y));
end

function x = swish(x)
    x = x./(1+exp(-x));
end