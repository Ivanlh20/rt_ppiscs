import numpy as np
import scipy.io as sio

def fcn_scs_fcc(x, crop, model):
    sz = x.shape
    if sz[1] != 9:
        x = np.transpose(x)

    x = np.float64(x)
    y = (x - model.x_sft) / model.x_sc
    y = fcn_eval_bz(y, model.bias, model.weights)
    if crop and (x[1,2] == 1):
        y = y[:, :61]
    y = y.T
    return y

def fcn_eval_bz(y_i, bias, weights, bz=4096*4):
    n_y = y_i.shape[0]
    if n_y <= bz:
        y = fcn_eval(y_i, bias, weights)
        return y
    nk = int(n_y/bz)
    y = np.zeros((n_y, len(bias[-1])))
    for ik in range(1, nk+1):
        ik_0 = (ik-1) * bz + 1
        ik_e = ik_0 + bz - 1
        y[ik_0:ik_e] = fcn_eval(y_i[ik_0:ik_e], bias, weights)
    if n_y > nk * bz:
        ik_0 = nk * bz + 1
        ik_e = n_y
        y[ik_0:ik_e] = fcn_eval(y_i[ik_0:ik_e], bias, weights)
    return y

def fcn_eval(y, bias, weights):
    n_lay = len(bias)
    y = swish(y * weights[0] + bias[0])
    for ik in range(1, n_lay-1):
        y = np.concatenate((y, swish(y * weights[ik] + bias[ik])))
    y = y * weights[n_lay] + bias[n_lay]
    y = np.log(1+np.exp(y))
    return y

def swish(x):
    x = x / (1 + np.exp(-x))
    return x

def ilp_scs_fcc(x, crop=True, model=None):
    sz = x.shape
    if ((sz[0] != 9) and (sz[1] != 9)):
        print('Wrong input dimensions')
        return
    if model is None:
        model = sio.loadmat('coef_scs_fcc.mat')
    y = fcn_scs_fcc(x, crop, model)
    return y