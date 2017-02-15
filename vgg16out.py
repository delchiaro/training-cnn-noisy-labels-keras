
# Command line application (and python function) that can compute the output of a trained vgg16 network over an images
# dataset (organized as: ./path/to/dataset/label/images.jpg) in a specific vgg16 layer.
# The computed output can be saved into an HDF5 file or as images in another folder.


from argparse import ArgumentParser

import sys

from nets.vgg16 import VGG16Builder, VGG16LayerBlock, VGG16Layers
from datasetLoader import ImageDataset

#-d ./dataset/4_ObjectCategories -lb ccp5 -hf 4_objCat_allimg_cpp5out.h5


# CUDA EXECUTE:
# THEANO_FLAGS=device=gpu,floatX=float32 python vgg16out.py -d ./dataset/4_ObjectCategories -lb ccp5 -hf 4_objCat_allimg_cpp5out.h5
OUTPUT_DATASET_NAME= 'data'
LABELS_DATASET_NAME= 'labels'
FILENAMES_DATASET_NAME= 'file_names'

DEFAULT_WEIGHTS_FILE="./weights/vgg16_weights.h5"
IMG_SIZE = [256, 256]
CROP_SIZE = [224, 224]


def vgg16out(img_dataset_path,
             weights_file=DEFAULT_WEIGHTS_FILE,
             output_layer=VGG16Layers.softmax,
             output_block=None,
             max_images_per_label=-1,
             hdf5_out_path='dataset_output.h5',
             img_out_path=None,
             shuffleBeforeSave=False):
    netbuilder = VGG16Builder()
    if output_block != None:
        output_layer = netbuilder.outputBlockToLayer(output_block)


    print("Loading image dataset from selected folder...")
    dataset = ImageDataset()
    dataset.loadImagesDataset(dataset_path=img_dataset_path, max_img_per_label=max_images_per_label, crop_size=CROP_SIZE, img_size=IMG_SIZE, sortFileNames=True)


    print("Building pretrained network from input to selected layer...")
    pretrained = netbuilder.pretrainedModel(weights_file, output_layer)

    print("Compute output of selected layer...")
    intermediate_output = pretrained.predict(dataset.data)
    # dataset.data=intermediate_output # unsafe !! (no shape/dimensions check)
    dataset.loadNumpyArray(intermediate_output, dataset.labels, dataset.fnames, dataset.labelmap )

    if shuffleBeforeSave:
        print("Shuffling dataset...")
        dataset.shuffle()
        dataset.shuffle()


    print("Writing output on h5 file...")
    dataset.saveToHDF5(hdf5_out_path)
    # h5f = h5py.File(hdf5_out_path, 'w')
    # h5f.create_dataset(OUTPUT_DATASET_NAME, data=intermediate_output)
    # h5f.create_dataset(LABELS_DATASET_NAME, data=dataset.labels)
    # h5f.create_dataset(FILENAMES_DATASET_NAME, data=dataset.fnames)
    # h5f.close()



    if img_out_path != None:
        print("Writing output as images...")
        dataset.saveToImages(img_out_path)



#def vgg16predict(imgParh)





def main(argv):

    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset", action='store', dest='dataset_path', required=True,
                        help="Specify the directory containing the dataset. <dataset_path>/<label>/<images_files.xxx>")

    parser.add_argument("-w", "--weights-file", action='store', dest='weights_file', default=DEFAULT_WEIGHTS_FILE,
                        help="Specify the hdf5 file containing the weights of the pretrained net. ")

    parser.add_argument("-ipl", "--images-per-label", action='store', dest='max_images_per_label', default=-1,
                        help="Specify the maximum number of images to load from the dataset per each label.")

    parser.add_argument("-l", #"--output-layer",
                        action='store', dest='output_layer',
                        default=VGG16Builder.getLayerNames()[-1], choices=VGG16Builder.getLayerNames(),
                        help="Select the output layer, using the layer name")

    parser.add_argument("-lb",  #"--output-layer-block",
                        action='store', dest='output_block',
                        default=None, choices=VGG16LayerBlock.keys(),
                        help="Select the network output layer, using the block name")

    parser.add_argument("-hf", "--hdf5-file", action='store', dest='hdf5_out_path', default='output.h5',
                        help = "Specify the path of the output h5 file.")

    parser.add_argument("-oi", "--output-image", action='store', dest='img_out_path', default=None,
                        help="Specify the path in which write the output as images. NOT YET IMPLEMENTED.")


    args=parser.parse_args()

    vgg16out(img_dataset_path = args.dataset_path,
             weights_file=args.weights_file,
             output_layer=args.output_layer,
             output_block=VGG16LayerBlock.buildFromString(args.output_block),
             hdf5_out_path=args.hdf5_out_path,
             max_images_per_label=args.max_images_per_label,
             img_out_path=args.img_out_path)



if __name__ == "__main__":
    main(sys.argv)




