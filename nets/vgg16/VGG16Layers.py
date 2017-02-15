


class VGG16Layers:
    # Simple container for all the layer names in the vgg-16 model.
    # NB: changing the layer string names will cause problem when loading the pretrained weights from the HDF5 file.

    input = 'input'
    conv1a = 'conv1_1'
    conv1b = 'conv1_2'
    pool1 = 'maxpooling2d_11'

    conv2a = 'conv2_1'
    conv2b = 'conv2_2'
    pool2 = 'maxpooling2d_12'

    conv3a = 'conv3_1'
    conv3b = 'conv3_2'
    conv3c = 'conv3_3'
    pool3 = 'maxpooling2d_13'

    conv4a = 'conv4_1'
    conv4b = 'conv4_2'
    conv4c = 'conv4_3'
    pool4 = 'maxpooling2d_14'

    conv5a = 'conv5_1'
    conv5b = 'conv5_2'
    conv5c = 'conv5_3'
    pool5 = 'maxpooling2d_15'

    flatten = 'flatten_3'
    dense1 = 'dense_1'
    drop1 = 'drop1'
    dense2 = 'dense_2'
    drop2 = 'drop2'
    dense3 = 'dense_3'
    softmax = 'softmax'