# Command line application (and python function) that can compute the output of a trained vgg16 network over an images
# dataset (organized as ./path/to/dataset/label/images.jpg or as ./path/to/dataset/images.jpg) or hdf5 dataset
# (see ImageDataset.saveToHDF5()) in a specific vgg16 layer.
# The computed output can be saved into an HDF5 file or as images in another folder.
# Refactoring of vgg16out.py


import os
from argparse import ArgumentParser

import h5py
import sys
from keras.models import Sequential, Model, model_from_json



#
# try:
#     os.chdir("/mnt/das4-fs4/")  # when executed locally
# except OSError as e:
#     os.chdir("/")  # when executed on das-4
# os.chdir("var/scratch/rdchiaro/")
from datasetLoader.ImageDataset import loadImageList, ImageDataset
from nets.vgg16 import VGG16Builder

DEFAULT_WEIGHTS_FILE = "/mnt/das4-fs4/var/scratch/rdchiaro/weights/vgg16_weights.h5"
DEFAULT_VGG16_INPUT_SIZE = [256, 256]
DEFAULT_VGG16_INPUT_CROP = [224, 224]


def file_extension(path):
    name, extension = os.path.splitext(path)
    return extension

# def hdf5_or_file(hdf5_in):
#     # type: ( var ) -> h5py
#     if isinstance(hdf5_in, str):
#         if os.path.isfile(hdf5_in):
#             input_hdf5 = h5py.File(hdf5_in, 'r')
#         else:
#             raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
#     elif not isinstance(hdf5_in, h5py):
#         raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
#     return input_hdf5

# def img_or_file(img):
#     # type: ( var ) -> h5py
#     if isinstance(hdf5_in, str):
#         if os.path.isfile(hdf5_in):
#             input_hdf5 = h5py.File(hdf5_in, 'r')
#         else:
#             raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
#     elif not isinstance(hdf5_in, h5py):
#         raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
#     return input_hdf5


def load_input(input, crop_size, img_size, load_folder_with_label = True):
    dataset = ImageDataset()
    if isinstance(input, str):
        if os.path.isfile(input):
            ext = file_extension(input)
            if ext == ".h5":
                #input_hdf5 = h5py.File(input, 'r')
                dataset.loadHDF5(input)

            elif ext == "":
                if load_folder_with_label:
                    dataset.loadImagesDataset(input, crop_size=crop_size, img_size=img_size, sortFileNames=True)
                else:
                    dataset.loadImagesFolder(input, crop_size=crop_size, img_size=img_size, sortFileNames=True)

            else:
                dataset.loadSingleImage(input, crop_size=crop_size, img_size=img_size)
        else:
            raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
    elif not isinstance(input, h5py.File):
        raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
    return dataset







def predict(input,
            model="vgg16",
            img_size=None,
            img_crop_size=None,
            weights=None,
            input_layer=None,
            output_layer=None,
            verbose=False):
    # #type: (hdf5, str, str, Model, str, str, str, str, bool) -> int
    # input: an image file, a folder containing images, a folder containing labe-folders containing images
    # model: a path to a keras json model, a name of a supported pre-build network (vgg16), a keras model

    if model is None:
        raise ValueError("model=None, should be a path to a json keras model, or a keras Model object")
    if isinstance(model, str):
        if os.path.isfile(model):
            model_json = open(model, "r")
            model = model_from_json(model_json)

        elif model == "vgg16":
            model = VGG16Builder.emptyModel()
            if weights is None:
                weights = DEFAULT_WEIGHTS_FILE
            if img_size is None:
                img_size = DEFAULT_VGG16_INPUT_SIZE
            if img_crop_size is None:
                img_size = DEFAULT_VGG16_INPUT_CROP
        else:
            raise ValueError("model json file not found or net name not supported")


    if input_layer is not None or output_layer is not None:
        model_input = model.input
        model_output = model.output
        if input_layer is not None:
            model.get_layer(input_layer).input
        if output_layer is not None:
            model.get_layer(output_layer).output
        model = Model(input=model_input, output=model_output)

    dataset = load_input(input, crop_size=img_crop_size, img_size=img_size)

    if weights is not None:
        if os.path.isfile(weights):
            weights = h5py.File(weights)
        if isinstance(weights, h5py.File):
            model.load_weights_from_hdf5_group_by_name(weights)
        else:
            raise ValueError("weights is not a valid hdf5 file or path to a valid hdf5 file.")


    prediction = model.predict(dataset.data, batch_size=len(dataset.data), verbose=True)
    if verbose:
        print prediction
    return prediction



def main(argv):

    parser = ArgumentParser()
    parser.add_argument('-i', action='store', dest='input', required=True)

    parser.add_argument("-is", "--img-size", action='store', dest='input_size', default=None,
                        help="Specify the size of the input")
    parser.add_argument("-ic", "--img-crop", action='store', dest='input_crop', default=None,
                        help="Specify input crop. ")

    parser.add_argument("-w", "--weights-file", action='store', dest='weights_file', default=DEFAULT_WEIGHTS_FILE,
                        help="Specify the hdf5 file containing the weights of the pretrained net. ")



    parser.add_argument("-n", "--net", action='store', dest='net', default="vgg16",
                        help="Specify the network model to use from the default network model available",
                        choices={"vgg16"})

    parser.add_argument("-nj", "--net-json", action='store', dest='net_json_file', default=None,
                        help="Specify the json file containing the network model definition.")

    parser.add_argument("-il", "--input-layer",
                        action='store', dest='input_layer', default=None,
                        help="Select the output layer, using the layer name")

    parser.add_argument("-ol", "--output-layer",
                        action='store', dest='output_layer', default=None,
                        help="Select the output layer, using the layer name")

    # parser.add_argument("-hf", "--hdf5-file", action='store', dest='hdf5_out_path', default='output.h5',
    #                     help="Specify the path of the output h5 file.")
    # parser.add_argument("-oi", "--output-image", action='store', dest='img_out_path', default=None,
    #                     help="Specify the path in which write the output as images. NOT YET IMPLEMENTED.")


    args=parser.parse_args()



    # if args.net_json_file is not None:
    #     if os.path.isfile(args.net_json_file):
    #         file = open(args.net_json_file, "r")
    #         jsnet = file.read()
    #         model = model_from_json(jsnet)
    if args.net is not None:
        model = args.net
    elif args.net_json_file is not None:
        model = args.net_json_file



    predict(input=args.input,
            model=model,
            img_size=args.input_size,
            img_crop_size=args.input_crop,
            weights=args.weights_file,
            input_layer=args.input_layer,
            output_layer=args.output_layer,
            verbose=True)



if __name__ == "__main__":
    main(sys.argv)




