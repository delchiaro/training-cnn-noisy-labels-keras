import errno
import numpy as np
import h5py as h5py
import sys

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.layers import Dense, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Optimizer
from keras.regularizers import l2

from Layers.LabelFlipNoise import LabelFlipNoise
from nets.vgg16 import VGG16Builder, VGG16LayerBlock
from datasetLoader import ImageDataset
from os import path
from nets.vgg16 import VGG16Layers
from vgg16out import vgg16out
import os


try: os.chdir("/mnt/das4-fs4/")  # when executed locally
except OSError as e: os.chdir("/")  # when executed on das-4
os.chdir("var/scratch/rdchiaro/")



IMAGES_DATASET_NOISY = '4_ObjectCategories_flip'
IMAGES_DATASET_TRUE = '4_ObjectCategories'
IMAGES_DATASET_PATH_NOISY = './dataset/' + IMAGES_DATASET_NOISY
IMAGES_DATASET_PATH_TRUE = './dataset/' + IMAGES_DATASET_TRUE

PRETRAIN_WEIGHTS_FILE = "./weights/vgg16_weights.h5"


FINE_TUNE            = True
CONF_MATRIX          = True
finetune_modelA      = False
finetune_modelA_true = False
finetune_modelC      = False
finetune_labelflip   = True


def main(args):
    # Common Options:
    MAX_IMG_PER_LABEL = -1
    dataset_noisy = load_dataset(IMAGES_DATASET_NOISY, IMAGES_DATASET_PATH_NOISY, MAX_IMG_PER_LABEL, PRETRAIN_WEIGHTS_FILE, VGG16Layers.input)
    dataset_true = load_dataset(IMAGES_DATASET_TRUE, IMAGES_DATASET_PATH_TRUE, MAX_IMG_PER_LABEL, PRETRAIN_WEIGHTS_FILE, VGG16Layers.input)

    VAL_SPLIT = 0.30
    BATCH_SIZE = 32


    if FINE_TUNE:
        if finetune_modelA_true:
            finetune_model(model_name="_A_TRUE", dataset=dataset_true, weight_file=PRETRAIN_WEIGHTS_FILE,
                           lr=0.001, decay=1e-6, momentum=0.9, nesterov=False,
                           epochs=20, val_split=VAL_SPLIT, batch_size=BATCH_SIZE,
                           label_flip=False, weight_decay=None,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

        if finetune_modelA:
            finetune_model(model_name="_A_NOISY", dataset=dataset_noisy, weight_file=PRETRAIN_WEIGHTS_FILE,
                           lr=0.001, decay=1e-6, momentum=0.9, nesterov=False,
                           epochs=2, val_split=VAL_SPLIT, batch_size=BATCH_SIZE,
                           label_flip=False, weight_decay=None,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=3)])



        ft_weights_file = "model_A_NOISY_after_tuning.h5"

        if finetune_labelflip:
            finetune_model(model_name="_FLIP_stochastic2", dataset=dataset_noisy, weight_file=ft_weights_file,
                           lr=0.01, decay=1e-5, momentum=0.5, nesterov=False,
                           epochs=10, val_split=VAL_SPLIT, batch_size=BATCH_SIZE,
                           label_flip=True, weight_decay=2)


        if finetune_modelC:
            finetune_model(model_name="_C", dataset=dataset_noisy, weight_file=ft_weights_file,
                           lr=0.01, decay=1e-5, momentum=0.5, nesterov=False,
                           epochs=10, val_split=VAL_SPLIT, batch_size=BATCH_SIZE,
                           label_flip=False, weight_decay=None,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=3)])


    if CONF_MATRIX:
        confusion_matrix(dataset_true, weight_file="model_A_TRUE_after_tuning.h5")
        confusion_matrix(dataset_true, weight_file="model_A_NOISY_after_tuning.h5")
        confusion_matrix(dataset_true, weight_file="model_C_after_tuning.h5")
        confusion_matrix(dataset_true, weight_file="model_FLIP_stochastic2_after_tuning.h5")
        # confusion_matrix(dataset_true, weight_file="model_FLIP_stochastic1_after_tuning.h5")
        # confusion_matrix(dataset_true, weight_file="model_FLIP_stochastic1_2_after_tuning.h5")
























def confusion_matrix(dataset, weight_file):
    if isinstance(weight_file, str):
        weights = h5py.File(weight_file, 'r')

    print ""
    print ""
    print ""
    print ("---- Confusion Matrix and data: " + weight_file)
    print ""

    sgd = SGD(lr=0, decay=0)
    model = get_model(weights, labelFlipNoiseLayer=False, optimizer=sgd)
    y_pred = model.predict(dataset.data)
    print y_pred
    y_pred = np.argmax(y_pred, axis=1)
    print y_pred

    target_names = [ '0_cars', '1_airplanes', '2_motorbikes', '3_faces']
    from sklearn.metrics import classification_report, confusion_matrix
    print( classification_report(dataset.getLabelsInt(), y_pred, target_names=target_names))
    print( confusion_matrix(dataset.getLabelsInt(), y_pred))














def finetune_model(model_name,               # type: str
                   dataset,                  # type: ImageDataset
                   weight_file,              # type: str
                   lr,                       # type: float
                   momentum,                 # type: float
                   decay,                    # type: float
                   nesterov,                 # type: bool
                   epochs,                   # type: int
                   val_split=0,              # type: int
                   batch_size=32,            # type: int
                   label_flip=False,         # type: bool
                   weight_decay=None,        # type: float
                   save_weights_before=True, # type: bool
                   save_weights_after=True,  # type: bool
                   callbacks=None            # type: list(Callback)
                   ):
    # type: (...) -> Model

    print ""
    print ("---- Fine Tuning model " + model_name)
    if isinstance(weight_file, str):
        weights = h5py.File(weight_file, 'r')
    print "EPOCHS       = " + str(epochs)
    print "BATCH SIZE   = " + str(batch_size)
    print "VALID SPLIT  = " + str(val_split * 100) + "%"
    print "------------"
    print "LEARN RATE   = " + str(lr)
    print "MOMENTUM     = " + str(momentum)
    print "LR DECAY     = "  + str(decay)
    print "NESTEROV     = " + str(nesterov)
    print "------------"
    print "LABEL FLIP   = " + str(label_flip)
    print "WEIGHT DECAY = " + str(weight_decay)
    print "-------------------------------------------"

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
    model = get_model(weights, labelFlipNoiseLayer=label_flip, weight_decay=weight_decay,
                      save_json_path="model{}.json".format(model_name), optimizer=sgd)

    if save_weights_before:
        model.save_weights("model{}_before_tuning.h5".format(model_name), overwrite=True)

    model.fit(x=dataset.data, y=dataset.labels,
              batch_size=batch_size, nb_epoch=epochs, validation_split=val_split, shuffle=True, callbacks=callbacks)

    if save_weights_after:
        model.save_weights("model{}_after_tuning.h5".format(model_name), overwrite=True)

    return model




















def get_model(weights, labelFlipNoiseLayer=False, weight_decay=None, optimizer=None, save_json_path = None):
    # type: (str, bool, float, Optimizer, str) -> Model
    trainable = not labelFlipNoiseLayer
    #trainable = True
    m1=0.0001
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224), name=VGG16Layers.input))
    model.add(Convolution2D(64, 3, 3, activation='relu', name=VGG16Layers.conv1a,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m1))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name=VGG16Layers.conv1b,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m1))  # H' = W' = 224
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool1))
    # W' = 112

    m2 = 0.005
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name=VGG16Layers.conv2a,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m2))  # W' = 112
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name=VGG16Layers.conv2b,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m2))  # W' = 112
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool2))
    # W' = 56

    m3 = 0.001
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name=VGG16Layers.conv3a,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m3))  # W' = 56
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name=VGG16Layers.conv3b,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m3))  # W' = 56
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name=VGG16Layers.conv3c,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m3))  # W' = 56
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool3))
    # W' = 28

    m4 = 0.1
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv4a,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m4))  # W' = 28
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv4b,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m4))  # W' = 28
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv4c,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m4))  # W' = 28
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool4))
    # W' = 14

    m5 = 0.5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv5a,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m5))  # W' = 14
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv5b,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m5))  # W' = 14
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name=VGG16Layers.conv5c,
                            W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m5))  # W' = 14
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name=VGG16Layers.pool5))
    # W' = 7


    m6=1
    # in: 512x49x1
    model.add(Flatten(name="flatten"))
    # out: 25088
    model.add(Dense(4096, activation='relu', name=VGG16Layers.dense1,
                    W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m6))
    model.add(Dropout(0.5, name=VGG16Layers.drop1))
    # fcd = fullyconnected-dropout # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
    # out: 4096x1x1           (512x49x4096 weights)

    m7=1
    # in: 4096x1x1
    model.add(Dense(4096, activation='relu', name=VGG16Layers.dense2,
                    W_learning_rate_multiplier=m1, b_learning_rate_multiplier=m7))
    model.add(Dropout(0.5, name=VGG16Layers.drop2))
    # Dropout is applied to Dropout input -> we are dropping out one of the 4096 neurons
    # out:  out: 4096x1x1      (4096x4096 weights ~=16M)

    # * * * * * * END PRETRAINED * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    m8 = 1
    model.add(Dense(4, name='dense_new', trainable=trainable, W_learning_rate_multiplier=m8, b_learning_rate_multiplier=m8))
    model.add(Activation("softmax", name="softmax"))

    mlf = 1
    if labelFlipNoiseLayer:
        if weight_decay is None:
            model.add(LabelFlipNoise(name='labelflip', W_learning_rate_multiplier=mlf))
        else:
            model.add(LabelFlipNoise(name='labelflip', W_learning_rate_multiplier=mlf, W_regularizer=l2(weight_decay)))

    if save_json_path:
        f = open(save_json_path, "w")
        f.write(model.to_json())

    if optimizer is not None:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("WARNING: Model not compiled (optimizer not provided).")
        print("         weights must be loaded AFTER model compiling, otherwise the weights will be initialized\n"
              "         from scratch, following the 'init' rule (default: random uniform distribution)")

    if weights is not None:
        model.load_weights_from_hdf5_group_by_name(weights)

    return model







def load_dataset(dataset_name, dataset_path,  max_img_per_label=-1, weight_file=None, output_layer=VGG16Layers.input):
    # type: (str, str, int, str, str) -> ImageDataset
    IPL = max_img_per_label
    if IPL < 0: IPL = "ALL"

    DATASET_HDF5_FILE = './dataset/' + dataset_name + '__VGG16__LAYER_' + output_layer + '__IPL_' + str(IPL) + '.h5'

    if path.isfile(DATASET_HDF5_FILE) == False:
        print(" - HDF5 dataset not found.. generating dataset from images dataset - ")
        vgg16out(img_dataset_path=dataset_path,
                 hdf5_out_path=DATASET_HDF5_FILE,
                 output_layer=output_layer,
                 max_images_per_label=max_img_per_label,
                 #output_block=OUTPUT_BLOCK,
                 weights_file=weight_file,
                 shuffleBeforeSave=True)

    dataset = ImageDataset()
    dataset.loadHDF5(DATASET_HDF5_FILE)
    # dataset.shuffle() no needs: shuffeled before saving
    return dataset






if __name__ == "__main__":
    main(sys.argv)

