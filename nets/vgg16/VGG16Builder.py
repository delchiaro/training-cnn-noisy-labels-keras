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
    def emptyModel(vgg16_last_layer=VGG16Layers.softmax):
        # type: (str, str) -> Model
        if vgg16_last_layer==VGG16Layers.input:
            submodel = Sequential()
            submodel.add(ZeroPadding2D((0, 0), input_shape=(3, 224, 224)))
        else:
            model = VGG16Builder._emptySequential()
            submodel = Model(input=model.input,
                             output=model.get_layer(vgg16_last_layer).output)
        return submodel

    @staticmethod
    def pretrainedModel(vgg16_weight_path, last_layer_name=VGG16Layers.softmax):
        # type: (str, str, str) -> Sequential
        weights = VGG16Builder._loadWeights(vgg16_weight_path)
        model = VGG16Builder.emptyModel(last_layer_name)
        model.load_weights_from_hdf5_group_by_name(weights)
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
    def _emptySequential():
        model = Sequential()

        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name=VGG16Layers.conv1a))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name=VGG16Layers.conv1b))  # H' = W' = 224
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool1))
        # W' = 112

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name=VGG16Layers.conv2a))  # W' = 112
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name=VGG16Layers.conv2b))  # W' = 112
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool2))
        # W' = 56

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name=VGG16Layers.conv3a))  # W' = 56
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name=VGG16Layers.conv3b))  # W' = 56
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name=VGG16Layers.conv3c))  # W' = 56
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool3))
        # W' = 28

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv4a))  # W' = 28
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv4b))  # W' = 28
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv4c))  # W' = 28
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool4))
        # W' = 14

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv5a))  # W' = 14
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv5b))  # W' = 14
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv5c))  # W' = 14
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool5))
        # W' = 7

        # in: 512x49x1
        model.add(Flatten(name="flatten"))
        # out: 25088
        model.add(Dense(4096, activation='relu', name=VGG16Layers.dense1))
        model.add(Dropout(0.5, name=VGG16Layers.drop1))
        # fcd = fullyconnected-dropout # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
        # out: 4096x1x1           (512x49x4096 weights)


        # in: 4096x1x1
        model.add(Dense(4096, activation='relu', name=VGG16Layers.dense2))
        model.add(Dropout(0.5, name=VGG16Layers.drop2))
        # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
        # out:  out: 4096x1x1      (4096x4096 weights ~=16M)

        # in: 4096x1x1
        model.add(Dense(1000, name=VGG16Layers.dense3))

        model.add(Activation("softmax", name=VGG16Layers.softmax))
        # out: 1000x1x1                 (1000x4096 weights ~= 4M)

        return model

    @staticmethod
    def _appendToSequential(model, layerIndex):
        if layerIndex == VGG16LayerBlock.ccp1.value:
            model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
            model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))  # H' = W' = 224
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))
            # W' = 112

        elif layerIndex == VGG16LayerBlock.ccp2.value:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))  # W' = 112
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))  # W' = 112
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))
            # W' = 56

        elif layerIndex == VGG16LayerBlock.ccp3.value:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))  # W' = 56
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))  # W' = 56
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))  # W' = 56
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))
            # W' = 28

        elif layerIndex == VGG16LayerBlock.ccp4.value:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))  # W' = 28
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))  # W' = 28
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))  # W' = 28
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))
            # W' = 14


        elif layerIndex == VGG16LayerBlock.ccp5.value:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))  # W' = 14
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))  # W' = 14
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))  # W' = 14
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))
            # W' = 7

        elif layerIndex == VGG16LayerBlock.fc1.value:
            # in: 512x49x1
            model.add(Flatten(name="flatten"))  # out: 25088
            model.add(Dense(4096, activation='relu', name='dense1'))
            model.add(
                Dropout(0.5))  # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
            # out: 4096x1x1           (512x49x4096 weights)

        elif layerIndex == VGG16LayerBlock.fc2.value:
            # in: 4096x1x1
            model.add(Dense(4096, activation='relu', name='dense2'))
            model.add(
                Dropout(0.5))  # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
            # out:  out: 4096x1x1      (4096x4096 weights ~=16M)

        elif layerIndex == VGG16LayerBlock.fc3.value:
            # in: 4096x1x1
            model.add(Dense(1000, name='dense3'))

        elif layerIndex == VGG16LayerBlock.sm.value:
            model.add(Activation("softmax", name="softmax"))
            # out: 1000x1x1                 (1000x4096 weights ~= 4M)

    @staticmethod
    def _loadWeights(vgg16_weight_path):
        import os.path
        if isinstance(vgg16_weight_path, basestring) == False:
            raise ValueError('vgg16_weight_path= must be of type "basestring"')
        elif os.path.isfile(vgg16_weight_path) == False:
            raise ValueError('vgg16_weight_path="' + vgg16_weight_path + '", file not exists')
        else:
            return h5py.File(vgg16_weight_path)







    # DEPRECATED METHODS:

    @deprecated
    @staticmethod
    def pretrainedSequential_byBlock(vgg16_weight_path, last_pretrained_layer_index=None):
        # type: ( str, VGG16LayerBlock) -> Sequential
        weights = VGG16Builder._loadWeights(vgg16_weight_path)
        model = VGG16Builder.emptySequential_byBlock(last_pretrained_layer_index)
        model.load_weights_from_hdf5_group_by_name(weights)
        return model

    @deprecated
    @staticmethod
    def emptyModel_byIndex(vgg16_last_layer=VGG16LayerBlock.sm):
        # type: (VGG16LayerBlock, VGG16LayerBlock) -> Model
        return VGG16Builder.emptyModel(VGG16Builder.outputBlockToLayer(vgg16_last_layer))

    @deprecated
    @staticmethod
    def pretrainedModel_byBlock(vgg16_weight_path, last_layer_block=VGG16LayerBlock.sm):
        # type: (str, VGG16LayerBlock, VGG16LayerBlock) -> Sequential
        weights = VGG16Builder._loadWeights(vgg16_weight_path)
        model = VGG16Builder.emptyModel_byIndex(last_layer_block)
        model.load_weights_from_hdf5_group_by_name(weights)
        return model

    @deprecated
    @staticmethod
    def printFilterMaps_byBlock(vgg16_weight_path, layer_block, output_dir_path, print_rgb_if_dim3=True):
        VGG16Builder.printFilterMaps(vgg16_weight_path, _filterMap[layer_block], output_dir_path, print_rgb_if_dim3)

    @deprecated
    @staticmethod
    def inputBlockToLayer(input_block):
        return _inputMap[input_block]

    @deprecated
    @staticmethod
    def outputBlockToLayer(output_block):
        return _outputMap[output_block]

    # TODO: BROKEN METHOD
    @deprecated
    @staticmethod
    def emptySequential_byBlock(last_pretrained_layer_block=None):  # type: (VGG16LayerBlock) -> Sequential
        if last_pretrained_layer_block == None:
            return VGG16Builder._emptySequential()
        else:
            model = Sequential()
            for i in range(1, last_pretrained_layer_block.value):
                VGG16Builder._appendToSequential(model, i)
            return model




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

_inputMap = {VGG16LayerBlock.ccp1: VGG16Layers.input,
             VGG16LayerBlock.ccp2: VGG16Layers.conv2a,
             VGG16LayerBlock.ccp3: VGG16Layers.conv3a,
             VGG16LayerBlock.ccp4: VGG16Layers.conv4a,
             VGG16LayerBlock.ccp5: VGG16Layers.conv5a,
             VGG16LayerBlock.fc1: VGG16Layers.dense1,
             VGG16LayerBlock.fc2: VGG16Layers.dense1,
             VGG16LayerBlock.fc3: VGG16Layers.dense3,
             VGG16LayerBlock.sm: VGG16Layers.softmax}

_outputMap = {VGG16LayerBlock.ccp1: VGG16Layers.pool1,
              VGG16LayerBlock.ccp2: VGG16Layers.pool2,
              VGG16LayerBlock.ccp3: VGG16Layers.pool3,
              VGG16LayerBlock.ccp4: VGG16Layers.pool4,
              VGG16LayerBlock.ccp5: VGG16Layers.pool5,
              VGG16LayerBlock.fc1: VGG16Layers.drop1,
              VGG16LayerBlock.fc2: VGG16Layers.drop2,
              VGG16LayerBlock.fc3: VGG16Layers.dense3,
              VGG16LayerBlock.sm: VGG16Layers.softmax}

_filterMap = {VGG16LayerBlock.ccp1: VGG16Layers.conv1b,
              VGG16LayerBlock.ccp2: VGG16Layers.conv2b,
              VGG16LayerBlock.ccp3: VGG16Layers.conv3c,
              VGG16LayerBlock.ccp4: VGG16Layers.conv4c,
              VGG16LayerBlock.ccp5: VGG16Layers.conv5c,
              VGG16LayerBlock.fc1: VGG16Layers.dense1,
              VGG16LayerBlock.fc2: VGG16Layers.dense2,
              VGG16LayerBlock.fc3: VGG16Layers.dense3,
              VGG16LayerBlock.sm: VGG16Layers.softmax}