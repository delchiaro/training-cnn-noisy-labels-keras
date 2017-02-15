import h5py as h5py
from keras.layers import Dense, Activation
from keras.layers import Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

from Layers.LabelFlipNoise import LabelFlipNoise
from nets.vgg16 import VGG16Builder, VGG16LayerBlock
from datasetLoader import ImageDataset
from os import path
from nets.vgg16 import VGG16Layers
from vgg16out import vgg16out

# Pretrained Output Options:
#OUTPUT_LAYER = VGG16Layers.pool5
OUTPUT_BLOCK = VGG16LayerBlock.ccp5
#IMAGES_DATASET= '4_ObjectCategories'
IMAGES_DATASET= '15_ObjectCategories'
IMAGES_DATASET_PATH = './dataset/' + IMAGES_DATASET
MAX_IMG_PER_LABEL = 2

# Fine-Tuning Options:
LR = 0.001
MOMENTUM = 0.9
DECAY = 1e-3
NESTEROV = True
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0
sgd = SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)

# Common Options:
WEIGHTS_FILE="./weights/vgg16_weights.h5"
IPL = MAX_IMG_PER_LABEL
if IPL < 0: IPL = "ALL"
DATASET_HDF5_FILE = './dataset/' + IMAGES_DATASET + '__VGG16__LAYER_' + VGG16Builder.outputBlockToLayer(OUTPUT_BLOCK)  +'__IPL_' + str(IPL) + '.h5'
IMG_SIZE = [256, 256]
CROP_SIZE = [224, 224]











if path.isfile(DATASET_HDF5_FILE) == False:
    print(" - HDF5 dataset not found.. generating dataset from images dataset - ")
    vgg16out(img_dataset_path=IMAGES_DATASET_PATH,
             hdf5_out_path=DATASET_HDF5_FILE,
             #output_layer=OUTPUT_LAYER,
             max_images_per_label=MAX_IMG_PER_LABEL,
             output_block=OUTPUT_BLOCK,
             weights_file=WEIGHTS_FILE)





trainable=True
netbuilder = VGG16Builder()
model = Sequential()
model.add(Flatten(input_shape=(512, 7, 7)))
model.add(Dense(4096, activation='relu', name='dense_1', trainable=trainable,
                W_learning_rate_multiplier=0.1, b_learning_rate_multiplier=0.1))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='dense_2', trainable=trainable,
                W_learning_rate_multiplier=0.5, b_learning_rate_multiplier=0.5))
model.add(Dropout(0.5))
model.load_weights_from_hdf5_group_by_name( h5py.File(WEIGHTS_FILE, 'r'))
model.add(Dense(6, name='dense_3', trainable=trainable,
                W_learning_rate_multiplier=1,  b_learning_rate_multiplier=1))

model.add(Activation("softmax", name="softmax"))

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# Idea: import python module passing name from program arguments, build the net model from that module
# var = "finetune_nets.vgg16_end_dd_1"
# lib = importlib.import_module(var, package=None)
# model = lib.getModel()



print "LR=" + str(LR)
print "MOMENTUM=" + str(MOMENTUM)
print "DECAY=" + str(DECAY)
print "EPOCHS=" + str(EPOCHS)
print "BATCH_SIZE=" + str(BATCH_SIZE)

dataset = ImageDataset()
dataset.loadHDF5(DATASET_HDF5_FILE)
dataset.shuffle()

# Training with automatic validation split
print "validation set = " + str(VALIDATION_SPLIT * 100) + "% of training set"
model.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)

# Training without automatic validation split:
# [trainX, testX] = np.split(X, 2)
# [trainY, testY] = np.split(Y, 2)
# print "trainX shape: " + str(trainX.shape)
# print "trainY shape: " + str(trainY.shape)
# model.fit(trainX, trainY, 8, 10 )
# print ""
# print"Evaluating the model on the test set: "
# print "testX shape: " + str(testX.shape)
# print "testY shape: " + str(testY.shape)
# ev = model.evaluate(testX, testY, 8)
# print ev

