import os
from os import makedirs

import h5py
import scipy
from keras.engine import Model
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from os.path import exists, join

from deprecation import deprecated
from nets.vgg16 import VGG16LayerBlock
from nets.vgg16 import VGG16Layers


from convnetskeras.customlayers import Softmax4D

_layers = [VGG16Layers.input,
           VGG16Layers.conv1a, VGG16Layers.conv1b,
           VGG16Layers.conv2a, VGG16Layers.conv2b,
           VGG16Layers.conv3a, VGG16Layers.conv3b, VGG16Layers.conv3c,
           VGG16Layers.conv4a, VGG16Layers.conv4b, VGG16Layers.conv4c,
           VGG16Layers.conv5a, VGG16Layers.conv5b, VGG16Layers.conv5c,
           VGG16Layers.flatten,
           VGG16Layers.dense1, VGG16Layers.drop1,
           VGG16Layers.dense2, VGG16Layers.drop2,
           VGG16Layers.dense3,
           VGG16Layers.softmax]


class VGG16Builder:
    # helper class to build vgg16 network.
    # Not a real need (we could simply use the methods of keras class 'Model' for loading model from json and weights
    # from hdf5) but has been built to fastly change the output layer and build keras Sequential submodel of VGG-16
    # (loading model from json and selecting a sublayer calling the constructor as
    # "submodel = Model(input=model.input, output=model.get_layer(vgg16_last_layer).output)"
    # will give an instance of the Model class, not Sequential).

    def __init__(self):
        pass



    @staticmethod
    def getLayerNames():
        return _layers


    @staticmethod
    def emptyModel(vgg16_last_layer=VGG16Layers.softmax, layer_dim=[], heatmap=False):
        # type: (str, str) -> Model
        if vgg16_last_layer==VGG16Layers.input and heatmap is False:
            submodel = Sequential()
            submodel.add(ZeroPadding2D((0, 0), input_shape=(3, 224, 224)))
        else:
            model = VGG16Builder._emptySequential(layer_dim=layer_dim, heatmap=heatmap)
            if vgg16_last_layer is not None and vgg16_last_layer is not VGG16Layers.softmax:
                model = Model(input=model.input, output=model.get_layer(vgg16_last_layer).output)
        return model


    @staticmethod
    def pretrainedModel(weights, last_layer_name=VGG16Layers.softmax,  layer_dim=[], heatmap=False):
        # type: (str, str, str) -> Sequential
        weights = VGG16Builder._loadWeights(weights)
        model = VGG16Builder.emptyModel(vgg16_last_layer=last_layer_name,  layer_dim=layer_dim, heatmap=False)
        model.load_weights_from_hdf5_group_by_name(weights)

        if heatmap:
            convnet_heatmap = VGG16Builder.emptyModel(vgg16_last_layer=last_layer_name,  layer_dim=layer_dim, heatmap=True)

            for layer in convnet_heatmap.layers:
                if layer.name.startswith("conv"):
                    orig_layer = model.get_layer(layer.name)
                    layer.set_weights(orig_layer.get_weights())
                elif layer.name.startswith("dense"):
                    orig_layer = model.get_layer(layer.name)
                    W, b = orig_layer.get_weights()
                    n_filter, previous_filter, ax1, ax2 = layer.get_weights()[0].shape
                    new_W = W.reshape((previous_filter, ax1, ax2, n_filter))
                    new_W = new_W.transpose((3, 0, 1, 2))
                    new_W = new_W[:, :, ::-1, ::-1]
                    layer.set_weights([new_W, b])
            return convnet_heatmap

        else:
            return model



    @staticmethod
    def printFilterMaps(vgg16_weight_path, layer_name, output_dir_path, print_rgb_if_dim3=True):
        model = VGG16Builder.pretrainedModel(vgg16_weight_path, layer_name)
        layer = model.get_layer(name=layer_name)
        weights = layer.get_weights()[0]
        # biases = layer.get_weights()[1]

        output_dir_path

        if weights is None:
            raise ValueError("Can't save to images: can't find weights in selected layer")

        if not exists(output_dir_path):
            makedirs(output_dir_path)

        index_filter = 0
        for filter in weights:
            filename = layer_name + '_filter_' + str(index_filter)
            if print_rgb_if_dim3 and filter.shape[0] == 3:
                fname = filename + '_rgb.jpg'
                scipy.misc.imsave(join(output_dir_path, fname), filter)

            else:
                index_map = 0
                for map in filter:
                    fname = filename + '_map_' + str(index_map) + '.jpg'
                    scipy.misc.imsave(join(output_dir_path, fname), map)
                    index_map += 1
            index_filter += 1



    @staticmethod
    def _emptySequential(layer_dim=[], heatmap=False):
        default_layer_dim = [1000, 4096, 4096, 512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64]
        layer_dim = layer_dim + default_layer_dim[len(layer_dim):]
        model = Sequential()


        if heatmap:
            model.add(ZeroPadding2D((1, 1), input_shape=(3, None, None)))
        else:
            model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))

        model.add(Convolution2D(layer_dim[-1], 3, 3, activation='relu', name=VGG16Layers.conv1a))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-2], 3, 3, activation='relu', name=VGG16Layers.conv1b))  # H' = W' = 224
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool1))
        # W' = 112

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-3], 3, 3, activation='relu', name=VGG16Layers.conv2a))  # W' = 112
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-4], 3, 3, activation='relu', name=VGG16Layers.conv2b))  # W' = 112
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool2))
        # W' = 56

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-5], 3, 3, activation='relu', name=VGG16Layers.conv3a))  # W' = 56
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-6], 3, 3, activation='relu', name=VGG16Layers.conv3b))  # W' = 56
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-7], 3, 3, activation='relu', name=VGG16Layers.conv3c))  # W' = 56
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool3))
        # W' = 28

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-8], 3, 3, activation='relu', name=VGG16Layers.conv4a))  # W' = 28
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-9], 3, 3, activation='relu', name=VGG16Layers.conv4b))  # W' = 28
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-10], 3, 3, activation='relu', name=VGG16Layers.conv4c))  # W' = 28
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool4))
        # W' = 14

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-11], 3, 3, activation='relu', name=VGG16Layers.conv5a))  # W' = 14
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-12], 3, 3, activation='relu', name=VGG16Layers.conv5b))  # W' = 14
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(layer_dim[-13], 3, 3, activation='relu', name=VGG16Layers.conv5c))  # W' = 14
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool5))
        # W' = 7

        if heatmap:
            model.add(Convolution2D(layer_dim[-14], 7, 7, activation="relu", name=VGG16Layers.dense1))
            model.add(Convolution2D(layer_dim[-15], 1, 1, activation="relu", name=VGG16Layers.dense2))
            model.add(Convolution2D(layer_dim[-16], 1, 1, name=VGG16Layers.dense3))
            model.add(Softmax4D(axis=1, name=VGG16Layers.softmax))
        else:

            # in: 512x49x1
            model.add(Flatten(name="flatten"))
            # out: 25088
            model.add(Dense(layer_dim[-14], activation='relu', name=VGG16Layers.dense1))
            model.add(Dropout(0.5, name=VGG16Layers.drop1))
            # fcd = fullyconnected-dropout # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
            # out: 4096x1x1           (512x49x4096 weights)


            # in: 4096x1x1
            model.add(Dense(layer_dim[-15], activation='relu', name=VGG16Layers.dense2))
            model.add(Dropout(0.5, name=VGG16Layers.drop2))
            # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
            # out:  out: 4096x1x1      (4096x4096 weights ~=16M)

            # in: 4096x1x1
            model.add(Dense(layer_dim[-16], name=VGG16Layers.dense3))
            model.add(Activation("softmax", name=VGG16Layers.softmax))
            # out: 1000x1x1                 (1000x4096 weights ~= 4M)

        return model

    @staticmethod
    def _loadWeights(weights):
        import os.path
        if isinstance(weights, basestring):
            if os.path.isfile(weights):
                weights = h5py.File(weights)
            elif os.path.isfile(weights) == False:
                raise ValueError('weights="' + weights + '", file not exists')
        return weights







    # DEPRECATED METHODS:
    #
    # @staticmethod
    # def _appendToSequential(model, layerIndex):
    #     if layerIndex == VGG16LayerBlock.ccp1.value:
    #         model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    #         model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))  # H' = W' = 224
    #         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #         # W' = 112
    #
    #     elif layerIndex == VGG16LayerBlock.ccp2.value:
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))  # W' = 112
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))  # W' = 112
    #         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #         # W' = 56
    #
    #     elif layerIndex == VGG16LayerBlock.ccp3.value:
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))  # W' = 56
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))  # W' = 56
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))  # W' = 56
    #         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #         # W' = 28
    #
    #     elif layerIndex == VGG16LayerBlock.ccp4.value:
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))  # W' = 28
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))  # W' = 28
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))  # W' = 28
    #         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #         # W' = 14
    #
    #
    #     elif layerIndex == VGG16LayerBlock.ccp5.value:
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))  # W' = 14
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))  # W' = 14
    #         model.add(ZeroPadding2D((1, 1)))
    #         model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))  # W' = 14
    #         model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #         # W' = 7
    #
    #     elif layerIndex == VGG16LayerBlock.fc1.value:
    #         # in: 512x49x1
    #         model.add(Flatten(name="flatten"))  # out: 25088
    #         model.add(Dense(4096, activation='relu', name='dense1'))
    #         model.add(
    #             Dropout(
    #                 0.5))  # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
    #         # out: 4096x1x1           (512x49x4096 weights)
    #
    #     elif layerIndex == VGG16LayerBlock.fc2.value:
    #         # in: 4096x1x1
    #         model.add(Dense(4096, activation='relu', name='dense2'))
    #         model.add(
    #             Dropout(
    #                 0.5))  # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
    #         # out:  out: 4096x1x1      (4096x4096 weights ~=16M)
    #
    #     elif layerIndex == VGG16LayerBlock.fc3.value:
    #         # in: 4096x1x1
    #         model.add(Dense(1000, name='dense3'))
    #
    #     elif layerIndex == VGG16LayerBlock.sm.value:
    #         model.add(Activation("softmax", name="softmax"))
    #         # out: 1000x1x1                 (1000x4096 weights ~= 4M)

                    # @deprecated
    # @staticmethod
    # def pretrainedSequential_byBlock(vgg16_weight_path, last_pretrained_layer_index=None):
    #     # type: ( str, VGG16LayerBlock) -> Sequential
    #     weights = VGG16Builder._loadWeights(vgg16_weight_path)
    #     model = VGG16Builder.emptySequential_byBlock(last_pretrained_layer_index)
    #     model.load_weights_from_hdf5_group_by_name(weights)
    #     return model
    #
    # @deprecated
    # @staticmethod
    # def emptyModel_byBlock(vgg16_last_layer=VGG16LayerBlock.sm):
    #     # type: (VGG16LayerBlock) -> Model
    #     return VGG16Builder.emptyModel(VGG16Builder.outputBlockToLayer(vgg16_last_layer))
    #
    # @deprecated
    # @staticmethod
    # def pretrainedModel_byBlock(vgg16_weight_path, last_layer_block=VGG16LayerBlock.sm):
    #     # type: (str, VGG16LayerBlock) -> Model
    #     weights = VGG16Builder._loadWeights(vgg16_weight_path)
    #     model = VGG16Builder.emptyModel_byBlock(last_layer_block)
    #     model.load_weights_from_hdf5_group_by_name(weights)
    #     return model
    #
    # @deprecated
    # @staticmethod
    # def printFilterMaps_byBlock(vgg16_weight_path, layer_block, output_dir_path, print_rgb_if_dim3=True):
    #     VGG16Builder.printFilterMaps(vgg16_weight_path, _filterMap[layer_block], output_dir_path, print_rgb_if_dim3)
    #
    # @deprecated
    # @staticmethod
    # def inputBlockToLayer(input_block):
    #     return _inputMap[input_block]
    #
    # @deprecated
    # @staticmethod
    # def outputBlockToLayer(output_block):
    #     return _outputMap[output_block]
    #
    # # TODO: BROKEN METHOD
    # @deprecated
    # @staticmethod
    # def emptySequential_byBlock(last_pretrained_layer_block=None):  # type: (VGG16LayerBlock) -> Sequential
    #     if last_pretrained_layer_block == None:
    #         return VGG16Builder._emptySequential()
    #     else:
    #         model = Sequential()
    #         for i in range(1, last_pretrained_layer_block.value):
    #             VGG16Builder._appendToSequential(model, i)
    #         return model



#
# _inputMap = {VGG16LayerBlock.ccp1: VGG16Layers.input,
#              VGG16LayerBlock.ccp2: VGG16Layers.conv2a,
#              VGG16LayerBlock.ccp3: VGG16Layers.conv3a,
#              VGG16LayerBlock.ccp4: VGG16Layers.conv4a,
#              VGG16LayerBlock.ccp5: VGG16Layers.conv5a,
#              VGG16LayerBlock.fc1: VGG16Layers.dense1,
#              VGG16LayerBlock.fc2: VGG16Layers.dense1,
#              VGG16LayerBlock.fc3: VGG16Layers.dense3,
#              VGG16LayerBlock.sm: VGG16Layers.softmax}
#
# _outputMap = {VGG16LayerBlock.ccp1: VGG16Layers.pool1,
#               VGG16LayerBlock.ccp2: VGG16Layers.pool2,
#               VGG16LayerBlock.ccp3: VGG16Layers.pool3,
#               VGG16LayerBlock.ccp4: VGG16Layers.pool4,
#               VGG16LayerBlock.ccp5: VGG16Layers.pool5,
#               VGG16LayerBlock.fc1: VGG16Layers.drop1,
#               VGG16LayerBlock.fc2: VGG16Layers.drop2,
#               VGG16LayerBlock.fc3: VGG16Layers.dense3,
#               VGG16LayerBlock.sm: VGG16Layers.softmax}
#
# _filterMap = {VGG16LayerBlock.ccp1: VGG16Layers.conv1b,
#               VGG16LayerBlock.ccp2: VGG16Layers.conv2b,
#               VGG16LayerBlock.ccp3: VGG16Layers.conv3c,
#               VGG16LayerBlock.ccp4: VGG16Layers.conv4c,
#               VGG16LayerBlock.ccp5: VGG16Layers.conv5c,
#               VGG16LayerBlock.fc1: VGG16Layers.dense1,
#               VGG16LayerBlock.fc2: VGG16Layers.dense2,
#               VGG16LayerBlock.fc3: VGG16Layers.dense3,
#               VGG16LayerBlock.sm: VGG16Layers.softmax}