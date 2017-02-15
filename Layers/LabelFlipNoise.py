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



import tensorflow as tf
import theano as th
# in future can be added to tensorflow_backend and theano_backend
def th_trace(x):
    return th.tensor.nlinalg.trace(x)
def tf_trace(x):
    return tf.trace(x)


class TraceRegularizer(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, alpha):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        regularization = 0
        if self.alpha:
            regularization += self.alpha * th_trace(x)
        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}


class LabelFlipNoise(Dense):

    def __init__(self,
                # weight_decay=0.1,
                 W_learning_rate_multiplier=None,
                 b_learning_rate_multiplier=None,
                 **kwargs):
        # Weight decay (regularizer_l2) is important because we want to keep the trace of the W matrix
        # (the matrix of the weight of this layer, i.e. the label-flip confusion matrix) low.
        # This because in the paper is proven that the tr(Q*) <= tr(Q), where Q* is the Q that best represents
        # the label-flip noise.

        Dense.__init__(self, output_dim=-1, bias=False, b_learning_rate_multiplier=None,
                       W_learning_rate_multiplier=W_learning_rate_multiplier,
                       #  W_regularizer=l2(weight_decay),
                       W_constraint=stochastic2(),
                       init='identity',
                       **kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.output_dim = self.input_dim = input_shape[-1]
        super(LabelFlipNoise, self).build(input_shape)

        # no more needed, because we are initializing the weights with init='identity'
        # W = self.get_weights()
        # W[0]=np.diag( np.ones(W[0].shape[0]))
        # self.set_weights(W)


    #
    # @Layer.trainable_weights.setter
    # def trainable_weights(self, weights):
    #     # TODO: transpose before transforming to stocMat?
    #     weights = matToStocMat(weights)  # we transform the matrix of weights in a stochastic matrix
    #     Layer.trainable_weights(self, weights)
    #
    # def set_weights(self, weights):
    #     # TODO: transpose before transforming to stocMat?
    #     weights = matToStocMat(weights)  # we transform the matrix of weights in a stochastic matrix
    #     Dense.set_weights(self, weights)

    def lock(self):
        trainable = setattr(self, 'trainable', False)

    def unlock(self):
        trainable = setattr(self, 'trainable', True)

    #TODO: do a version of this layer with an integrated cross entropy cost,
    #       or manage that the next layer must be a loss layer and that,
    #       when the loss derivate is low, this layer unlocks automatically
    #       to absorb the label-flip noise.













class Stochastic(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.

    # Arguments
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Convolution2D` layer with `dim_ordering="tf"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):

        min = K.min(p, axis=self.axis, keepdims=True)
        p = p - min * K.cast(min < 0., K.floatx())
        max = K.max(p, axis=self.axis, keepdims=True)
        p = p/max
        return p / (K.epsilon() + K.sum(p,axis=self.axis,keepdims=True))
        # return p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis,keepdims=True)))


    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}



class UnitNormNonNeg(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.

    # Arguments
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Convolution2D` layer with `dim_ordering="tf"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p / (K.epsilon() + K.sum(p,axis=self.axis,keepdims=True))
        #return p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}

stochastic = Stochastic
stochastic2 = UnitNormNonNeg
