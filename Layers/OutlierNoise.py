from keras.constraints import unitnorm, Constraint
from keras.engine import InputSpec
from keras.engine import Layer
from keras.layers import Dense, initializations
from keras import backend as K

import numpy as np
from keras.regularizers import l2, Regularizer

#
# def vecToStocVec(v):
#     if v.min() < 0:
#         v -= v.min()
#     v /= v.sum()
#     return v
#
# def matToStocMat(mat):
#     for v in mat:
#         v = vecToStocVec(v)
#     return mat




class OutlierNoise(Dense):

    def __init__(self,
                 alpha=1,
                 **kwargs):
        # Weight decay (regularizer_l2) is important because we want to keep the trace of the W matrix
        # (the matrix of the weight of this layer, i.e. the label-flip confusion matrix) low.
        # This because in the paper is proven that the tr(Q*) <= tr(Q), where Q* is the Q that best represents
        # the label-flip noise.

        Dense.__init__(self, output_dim=-1, bias=False, trainable=False, init='identity', **kwargs)
        if alpha >= 1:
            raise ValueError("OutlierNoise Layer: alpha must be < 1 "
                             "(theorically alpha = (outlaiers labelled as outlaier i.e. class K+1) / (total outlaiers)")
        self.alpha = alpha

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.output_dim = self.input_dim = input_shape[-1]
        super(OutlierNoise, self).build(input_shape)

        K = self.input_dim - 1
        W = self.get_weights()
        W[0]=np.diag( np.ones(W[0].shape[0]))

        # setting last column to (1-alpha)/K  (or row???)
        for i in range(0, self.output_dim-1):
            W[0][i][-1] = (1-self.alpha)/K

        # setting last row to (1-alpha)/K  (or column???)
        #W[0][-1] = np.full([self.output_dim], (1-self.alpha)/K)


        W[0][-1][-1] = self.alpha
        self.set_weights(W)

    def lock(self):
        trainable = setattr(self, 'trainable', False)

    def unlock(self):
        trainable = setattr(self, 'trainable', True)
