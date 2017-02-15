
from enum import Enum

from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation



class VGG16LayerBlock(Enum):
    ccp1 = 1 #  3   x 224 x 224 ->  Conv(relu) -> Conv(relu) -> pool  ->    64   x 112 x 112
    ccp2 = 2 #  64  x 112 x 112 ->  Conv(relu) -> Conv(relu) -> pool  ->    128  x 56  x  56
    ccp3 = 3 #  128 x 56  x 56  ->  Conv(relu) -> Conv(relu) -> pool  ->    256  x 28  x  28
    ccp4 = 4 #  256 x 28  x 28  ->  Conv(relu) -> Conv(relu) -> pool  ->    512  x 14  x  14
    ccp5 = 5 #  512 x 14  x 14  ->  Conv(relu) -> Conv(relu) -> pool  ->    1024 x  7  x   7
    fc1 = 6 #  1024 x 7 x 7 -> FLATTEN -> 1024 x 49 x 1 -> FullyConnected -> Relu ->  4096 x 1 x 1
    fc2 = 7 #  4096 x 1 x 1 -> FullyConnected -> Relu ->  4096 x 1 x 1
    fc3 = 8 #  4096 x 1 x 1 -> FullyConnected -> Relu ->  1000 x 1 x 1d
    sm = 9  #  1000 x 1 x 1 -> Softmax Activation ->  1000 x 1 x 1 stochastic vector

    def getName(self):
        return VGG16LayerBlock.ccp1.keys()[self.value-1]

    @staticmethod
    def dict():
        return VGG16LayerBlock.__members__

    @staticmethod
    def keys():
        return VGG16LayerBlock.__members__.keys()

    @staticmethod
    def buildFromString(block_name):
        return VGG16LayerBlock.dict()[block_name]

