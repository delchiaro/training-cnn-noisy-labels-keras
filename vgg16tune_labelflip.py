import errno
import h5py as h5py
from keras.layers import Dense, Activation
from keras.layers import Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2

from Layers.LabelFlipNoise import LabelFlipNoise
from nets.vgg16 import VGG16Builder, VGG16LayerBlock
from datasetLoader import ImageDataset
from os import path
from nets.vgg16 import VGG16Layers
from vgg16out import vgg16out
import os

# Pretrained Output Options:
#OUTPUT_LAYER = VGG16Layers.pool5
OUTPUT_BLOCK = VGG16LayerBlock.ccp5
#IMAGES_DATASET= '4_ObjectCategories'
IMAGES_DATASET= '15_ObjectCategories_labelflip2'
IMAGES_DATASET_PATH = './dataset/' + IMAGES_DATASET
MAX_IMG_PER_LABEL = -1

# Fine-Tuning Options:
LR = 0.001
MOMENTUM = 0.8
DECAY = 1e-3
NESTEROV = False
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.40
sgd = SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)

# Common Options:
WEIGHTS_FILE="./weights/vgg16_weights.h5"
IPL = MAX_IMG_PER_LABEL
if IPL < 0: IPL = "ALL"
DATASET_HDF5_FILE = './dataset/' + IMAGES_DATASET + '__VGG16__LAYER_' + VGG16Builder.outputBlockToLayer(OUTPUT_BLOCK)  +'__IPL_' + str(IPL) + '.h5'
IMG_SIZE = [256, 256]
CROP_SIZE = [224, 224]

try:
    os.chdir("/mnt/das4-fs4/")  # when executed locally
except OSError as e:
    os.chdir("/")  # when executed on das-4

os.chdir("var/scratch/rdchiaro/")







def getModel(weights, labelFlipNoiseLayer=False, weight_decay=None):

    netbuilder = VGG16Builder()
    trainable = not labelFlipNoiseLayer
    trainable = True
    model = Sequential()
    model.add(Flatten(input_shape=(512, 7, 7)))
    model.add(Dense(4096, activation='relu', name='dense_1', trainable=trainable, W_learning_rate_multiplier=0.1, b_learning_rate_multiplier=0.1))
    if trainable: model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2', trainable=trainable,W_learning_rate_multiplier=0.5, b_learning_rate_multiplier=0.5))
    if trainable: model.add(Dropout(0.5))
    model.add(Dense(6, name='dense_new', trainable=trainable, W_learning_rate_multiplier=1, b_learning_rate_multiplier=1))
    model.add(Activation("softmax", name="softmax"))

    model.load_weights_from_hdf5_group_by_name(weights)

    if labelFlipNoiseLayer:
        if weight_decay is None:
            model.add(LabelFlipNoise(name='labelflip', W_learning_rate_multiplier=2))
        else:
            model.add(LabelFlipNoise(name='labelflip', W_learning_rate_multiplier=2, W_regularizer=l2(weight_decay)))

    return model

if path.isfile(DATASET_HDF5_FILE) == False:
    print(" - HDF5 dataset not found.. generating dataset from images dataset - ")
    vgg16out(img_dataset_path=IMAGES_DATASET_PATH,
             hdf5_out_path=DATASET_HDF5_FILE,
             #output_layer=OUTPUT_LAYER,
             max_images_per_label=MAX_IMG_PER_LABEL,
             output_block=OUTPUT_BLOCK,
             weights_file=WEIGHTS_FILE)




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

modelA = getModel(h5py.File(WEIGHTS_FILE, 'r'), False)
modelA.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
modelA.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
modelA.save_weights("modelA.h5")
#loss: 0.0765 - acc: 0.9770

fit = modelA.evaluate(x=dataset.data, y=dataset.labels, batch_size=467)
print ''
print fit





weights = h5py.File("modelA.h5", 'r')
LR = 0.00001
MOMENTUM = 0.3
DECAY = 1e-7
NESTEROV = False
EPOCHS = 20
sgd = SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)

print "LR=" + str(LR)
print "MOMENTUM=" + str(MOMENTUM)
print "DECAY=" + str(DECAY)
print "EPOCHS=" + str(EPOCHS)
print "BATCH_SIZE=" + str(BATCH_SIZE)
#
# modelC1 = getModel(weights, False)
# modelC1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# modelC1.save_weights("modelC1_pretrained.h5")
# modelC1.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
# modelC1.save_weights("modelC1_trained.h5")
#
#
# modelC2 = getModel(weights, False)
# modelC2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# modelC2.save_weights("modelC2_pretrained.h5")
# modelC2.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
# modelC2.save_weights("modelC2_trained.h5")
#
#
# modelC3 = getModel(weights, False)
# modelC3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# modelC3.save_weights("modelC3_pretrained.h5")
# modelC3.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
# modelC3.save_weights("modelC3_trained.h5")
#
#
# print "\nB1:"
# modelB1 = getModel(weights, True)
# modelB1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# modelB1.save_weights("modelB1_pretrained.h5")
# modelB1.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
# modelB1.save_weights("modelB1_trained.h5")

print "\nB2:"
modelB2 = getModel(weights, True, 0.8)
modelB2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
modelB2.save_weights("modelB2_pretrained.h5")
modelB2.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
modelB2.save_weights("modelB2_trained.h5")
fit = modelB2.evaluate(x=dataset.data, y=dataset.labels, batch_size=467)
print ''
print fit
#
# print "\nB3:"
# modelB3 = getModel(weights, True, 0.05)
# modelB3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# modelB3.save_weights("modelB3_pretrained.h5")
# modelB3.fit(x=dataset.data, y=dataset.labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, shuffle=True)
# modelB3.save_weights("modelB3_trained.h5")
# modelB3 = getModel(weights, True)
#
