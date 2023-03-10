import numpy as np
import scipy.io as sio

class PPISCS_Model:
    def __init__(self, model_path='coef_scs_fcc.mat'):
        self._model = sio.loadmat(model_path)['model'][0, 0]
        self._bias = [np.array(x) for x in self._model['bias'][0]]
        self._weights = [np.array(x) for x in self._model['weights'][0]]

    @staticmethod
    def swish(x):
        return x / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)

    @staticmethod
    def eval_on_batch(y, layer_bias, layer_weights):
        n_layers = len(layer_bias)
        y = PPISCS_Model.swish(np.matmul(y, layer_weights[0]) + layer_bias[0])
        for i in range(1, n_layers - 1):
            y = np.hstack((y, PPISCS_Model.swish(np.matmul(y, layer_weights[i]) + layer_bias[i])))
        y = np.matmul(y, layer_weights[-1]) + layer_bias[-1]
        return PPISCS_Model.softplus(y)

    @staticmethod
    def eval(y_i, bias, weights, batch_size):
        n_y = y_i.shape[0]
        if n_y <= batch_size:
            y_out = PPISCS_Model.eval_on_batch(y_i, bias, weights)
            return y_out

        y_out = np.empty((n_y, bias[-1].size), dtype=y_i.dtype)
        for ik in range(0, n_y, batch_size):
            y_out[ik:ik + batch_size] = PPISCS_Model.eval_on_batch(y_i[ik:ik + batch_size], bias, weights)
        return y_out

    def predict(self, input_data, batch_size=4096*4, crop=True):
        if not isinstance(input_data, np.ndarray) or input_data.ndim != 2 or input_data.shape[1] != 9:
            raise ValueError('Input data must be a 2D numpy array with 9 columns')
        if crop and input_data[0, 1] == 1:
            crop_end = 61
        else:
            crop_end = None
        y = (input_data.astype(np.float64)- self._model['x_sft']) / self._model['x_sc']
        y = PPISCS_Model.eval(y, self._bias, self._weights, batch_size)
        if crop_end is not None:
            y = y[:, :crop_end]
        return y