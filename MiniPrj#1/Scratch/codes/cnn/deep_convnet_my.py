# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from common.layers import *


class BatchNormalization:
    """Batch Normalization

    FC 입력(N, D)과 CNN 입력(N, C, H, W)을 모두 지원한다.
    CNN 입력은 (N, C, H, W) -> (N*H*W, C)로 바꿔 채널별 BN을 적용한다.
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.running_mean = running_mean
        self.running_var = running_var
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape

        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        out = self.__forward(x, train_flg)

        if len(self.input_shape) != 2:
            N, C, H, W = self.input_shape
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return out

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)

        dx = self.__backward(dout)

        if len(self.input_shape) != 2:
            N, C, H, W = self.input_shape
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx



class DeepConvNet:
    """6개 합성곱층 + BatchNorm 유지 버전

    구조:
        conv - bn - relu - conv - bn - relu - dropout - pool -
        conv - bn - relu - dropout - conv - bn - relu - dropout - pool -
        conv - bn - relu - conv - bn - relu - pool -
        affine - bn - relu - dropout - affine - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2={'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3={'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4={'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10,
                 use_batchnorm=True):

        self.use_batchnorm = use_batchnorm
        conv_params = [conv_param_1, conv_param_2, conv_param_3,
                       conv_param_4, conv_param_5, conv_param_6]

        def conv_output_size(size, conv_param):
            return int((size + 2 * conv_param['pad'] - conv_param['filter_size']) /
                       conv_param['stride'] + 1)

        def pool_output_size(size, pool_size=2, stride=2, pad=0):
            return int((size + 2 * pad - pool_size) / stride + 1)

        # 가중치 초기화용 fan-in 계산
        pre_node_nums = []
        pre_channel_num = input_dim[0]
        for conv_param in conv_params:
            filter_size = conv_param['filter_size']
            pre_node_nums.append(pre_channel_num * filter_size * filter_size)
            pre_channel_num = conv_param['filter_num']

        # 두 번째 코드 구조처럼 conv2, conv4, conv6 뒤에 pooling
        feature_map_size = input_dim[1]
        for idx, conv_param in enumerate(conv_params):
            feature_map_size = conv_output_size(feature_map_size, conv_param)
            if idx in (1, 3, 5):
                feature_map_size = pool_output_size(feature_map_size)

        affine_input_size = pre_channel_num * feature_map_size * feature_map_size
        pre_node_nums.append(affine_input_size)
        pre_node_nums.append(hidden_size)

        weight_init_scales = np.sqrt(2.0 / np.array(pre_node_nums))

        self.params = {}

        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(conv_params):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(
                conv_param['filter_num'],
                pre_channel_num,
                conv_param['filter_size'],
                conv_param['filter_size']
            )
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.params['W7'] = weight_init_scales[6] * np.random.randn(
            affine_input_size, hidden_size
        )
        self.params['b7'] = np.zeros(hidden_size)

        self.params['W8'] = weight_init_scales[7] * np.random.randn(
            hidden_size, output_size
        )
        self.params['b8'] = np.zeros(output_size)

        if self.use_batchnorm:
            for idx, conv_param in enumerate(conv_params):
                self.params['gamma' + str(idx+1)] = np.ones(conv_param['filter_num'])
                self.params['beta' + str(idx+1)] = np.zeros(conv_param['filter_num'])

            self.params['gamma7'] = np.ones(hidden_size)
            self.params['beta7'] = np.zeros(hidden_size)

        self.layers = []

        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma1'], self.params['beta1']))
        self.layers.append(Relu())

        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma2'], self.params['beta2']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.1))
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma3'], self.params['beta3']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.1))

        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma4'], self.params['beta4']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.2))
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma5'], self.params['beta5']))
        self.layers.append(Relu())

        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma6'], self.params['beta6']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma7'], self.params['beta7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.3))

        self.layers.append(Affine(self.params['W8'], self.params['b8']))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        grads = {}

        if self.use_batchnorm:
            weight_layer_idx = (0, 3, 8, 12, 17, 20, 24, 28)
            bn_layer_idx = (1, 4, 9, 13, 18, 21, 25)

            for i, layer_idx in enumerate(weight_layer_idx):
                grads['W' + str(i+1)] = self.layers[layer_idx].dW
                grads['b' + str(i+1)] = self.layers[layer_idx].db

            for i, layer_idx in enumerate(bn_layer_idx):
                grads['gamma' + str(i+1)] = self.layers[layer_idx].dgamma
                grads['beta' + str(i+1)] = self.layers[layer_idx].dbeta

        else:
            weight_layer_idx = (0, 2, 6, 9, 13, 15, 18, 21)

            for i, layer_idx in enumerate(weight_layer_idx):
                grads['W' + str(i+1)] = self.layers[layer_idx].dW
                grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        with open(os.path.dirname(__file__) + '/' + file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        if self.use_batchnorm:
            weight_layer_idx = (0, 3, 8, 12, 17, 20, 24, 28)
            bn_layer_idx = (1, 4, 9, 13, 18, 21, 25)

            for i, layer_idx in enumerate(weight_layer_idx):
                self.layers[layer_idx].W = self.params['W' + str(i+1)]
                self.layers[layer_idx].b = self.params['b' + str(i+1)]

            for i, layer_idx in enumerate(bn_layer_idx):
                self.layers[layer_idx].gamma = self.params['gamma' + str(i+1)]
                self.layers[layer_idx].beta = self.params['beta' + str(i+1)]

        else:
            weight_layer_idx = (0, 2, 6, 9, 13, 15, 18, 21)

            for i, layer_idx in enumerate(weight_layer_idx):
                self.layers[layer_idx].W = self.params['W' + str(i+1)]
                self.layers[layer_idx].b = self.params['b' + str(i+1)]
