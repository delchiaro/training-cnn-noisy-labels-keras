# Command line application (and python function) that can compute the output of a trained vgg16 network over an images
# dataset (organized as ./path/to/dataset/label/images.jpg or as ./path/to/dataset/images.jpg) or hdf5 dataset
# (see ImageDataset.saveToHDF5()) in a specific vgg16 layer.
# The computed output can be saved into an HDF5 file or as images in another folder.
# Refactoring of vgg16out.py


import os
from argparse import ArgumentParser

import h5py
import sys

from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import SGD
from theano.gradient import np

from datasetLoader.ImageDataset import loadImageList, ImageDataset
from nets.vgg16 import VGG16Builder

DEFAULT_WEIGHTS_FILE = "/mnt/das4-fs4/var/scratch/rdchiaro/weights/vgg16_weights.h5"
DEFAULT_VGG16_INPUT_SIZE = [256, 256]
DEFAULT_VGG16_INPUT_CROP = [224, 224]


def file_extension(path):
    name, extension = os.path.splitext(path)
    return extension





def predict(input_image, layer_dim=[], weights=None, verbose=True):
    # input: an image file, a folder containing images, a folder containing labe-folders containing images
    # model: a path to a keras json model, a name of a supported pre-build network (vgg16), a keras model


    if weights is None:
        weights = DEFAULT_WEIGHTS_FILE
    elif isinstance(weights, basestring) and os.path.isfile(weights):
            weights = h5py.File(weights)

    if not isinstance(weights, h5py.File):
        raise ValueError("weights is not a valid hdf5 file or path to a valid hdf5 file.")


    im = preprocess_image_batch([input_image], color_mode="rgb")

    model = VGG16Builder.pretrainedModel(weights=weights, layer_dim=layer_dim, heatmap=True)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse')

    prediction = model.predict(im)


    if verbose:
        print prediction
    return prediction


if __name__ == "__main__":

    try:
        os.chdir("/mnt/das4-fs4/")  # when executed locally
    except OSError as e:
        os.chdir("/")  # when executed on das-4
    os.chdir("var/scratch/rdchiaro/")

    # TODO: NOT WORKING WITH VGG-16 !! :(
    # predict('car.jpg', layer_dim=[], weights='weights/vgg16_weights.h5')
    #


    # TODO: NOT WORKING WITH VGG-16 !! :(
    s = "n02084071"
    ids = synset_to_dfs_ids(s)
    # Most of the synsets are not in the subset of the synsets used in ImageNet recognition task.
    ids = np.array([id_ for id_ in ids if id_ is not None])

    im = preprocess_image_batch(['dog.jpg'], color_mode="rgb")

    # Test pretrained model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=True)
    model.compile(optimizer=sgd, loss='mse')


    out = model.predict(im)
    s = "n02084071"
    ids = synset_to_dfs_ids(s)
    print ids
    heatmap = out[0, ids].sum(axis=0)
    print heatmap
    # Then, we can get the image
    import matplotlib.pyplot as plt

    plt.imsave("heatmap_dog.png", heatmap)

    h5f = h5py.File("heatmap_dog.h5", 'w')
    h5f.create_dataset("heatmap", data=heatmap)
    h5f.close()
#
# def main(argv):
#
#     parser = ArgumentParser()
#     parser.add_argument('-i', action='store', dest='input', required=True)
#
#     parser.add_argument("-is", "--img-size", action='store', dest='input_size', default=None,
#                         help="Specify the size of the input")
#     parser.add_argument("-ic", "--img-crop", action='store', dest='input_crop', default=None,
#                         help="Specify input crop. ")
#
#     parser.add_argument("-w", "--weights-file", action='store', dest='weights_file', default=DEFAULT_WEIGHTS_FILE,
#                         help="Specify the hdf5 file containing the weights of the pretrained net. ")
#
#
#
#     parser.add_argument("-n", "--net", action='store', dest='net', default="vgg16",
#                         help="Specify the network model to use from the default network model available",
#                         choices={"vgg16"})
#
#     parser.add_argument("-nj", "--net-json", action='store', dest='net_json_file', default=None,
#                         help="Specify the json file containing the network model definition.")
#
#     parser.add_argument("-il", "--input-layer",
#                         action='store', dest='input_layer', default=None,
#                         help="Select the output layer, using the layer name")
#
#     parser.add_argument("-ol", "--output-layer",
#                         action='store', dest='output_layer', default=None,
#                         help="Select the output layer, using the layer name")
#
#     # parser.add_argument("-hf", "--hdf5-file", action='store', dest='hdf5_out_path', default='output.h5',
#     #                     help="Specify the path of the output h5 file.")
#     # parser.add_argument("-oi", "--output-image", action='store', dest='img_out_path', default=None,
#     #                     help="Specify the path in which write the output as images. NOT YET IMPLEMENTED.")
#
#
#     args=parser.parse_args()
#
#
#
#     # if args.net_json_file is not None:
#     #     if os.path.isfile(args.net_json_file):
#     #         file = open(args.net_json_file, "r")
#     #         jsnet = file.read()
#     #         model = model_from_json(jsnet)
#     if args.net is not None:
#         model = args.net
#     elif args.net_json_file is not None:
#         model = args.net_json_file
#
#
#
#     predict(input=args.input,
#             model=model,
#             img_size=args.input_size,
#             img_crop_size=args.input_crop,
#             weights=args.weights_file,
#             input_layer=args.input_layer,
#             output_layer=args.output_layer,
#             verbose=True)




