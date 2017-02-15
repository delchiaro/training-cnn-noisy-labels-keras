


from argparse import ArgumentParser

import h5py as h5py

from nets.vgg16 import VGG16Builder, VGG16LayerBlock, VGG16Layers
from datasetLoader import ImageDataset
from os import path

DEFAULT_WEIGHTS_FILE="./weights/vgg16_weights.h5"
IMG_SIZE = [256, 256]
CROP_SIZE = [224, 224]

netbuilder = VGG16Builder()

parser = ArgumentParser()



# parser.add_argument("-n", "--net", action='store', dest='net',
#                     default='vgg16', choices=['vgg16', 'alexnet'],
#                     help="Select the network model to use")
# parser.add_argument("-wd", "--weights-directory", action='store', dest='weights_dir', default='./weights/',
#                     help="Specify directory of the hdf5 file containing the weights of the pretrained net")

parser.add_argument("-l", "--layer", action='store', dest='layer', required=False, default=VGG16Layers.conv1a,
                    choices=netbuilder.getLayerNames() ,
                    help="Select the layer from which to extract the filters")

parser.add_argument("-lb", "--layer-block", action='store', dest='layer_block', required=False, default=None,
                    choices=VGG16LayerBlock.keys() ,
                    help="Select the layer-block from which to extract the filters")

parser.add_argument("-w", "--weights-file", action='store', dest='weights_file', default=DEFAULT_WEIGHTS_FILE,
                    help="Specify the hdf5 file containing the weights of the pretrained net. ")

parser.add_argument("-o", "--output", action='store', dest='img_out_path', default="./filter_maps",
                    help="Specify the path in which write the output as images.")


#
# parser.add_argument("-hf", "--hdf5-file", action='store', dest='hdf5_out_path', default='output.h5',
#                     help = "Specify the path of the output h5 file.")


args=parser.parse_args()


if args.layer_block != None:
    netbuilder.printFilterMaps_byBlock(args.weights_file, args.layer_block, args.img_out_path)

else:
    netbuilder.printFilterMaps(args.weights_file, args.layer, args.img_out_path)




