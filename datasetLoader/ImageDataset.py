import ntpath
from os.path import join, exists
from os import makedirs

import h5py
import scipy
from scipy.misc import imread, imresize, imsave, imshow
import numpy as np

DEFAULT_EXTENSIONS=[".jpg", ".jpeg", ".png", ".gif", ".bmp"]
DATA_DATASET_NAME= 'data'
LABELS_DATASET_NAME= 'labels'
# SECONDARY_LABEL_DATASET_NAME= 'secondary_labels'

LABELS_MAP_NAME= 'labels_map'
FILENAMES_DATASET_NAME= 'file_names'


def _path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def _listOnlyDirs(path):
    from os import listdir
    from os.path import isfile, join
    return  [f for f in listdir(path) if (isfile(join(path, f)) == False)]

def _listOnlyFiles(path):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(path) if isfile(join(path, f))]

def _listOnlyFilesWithExt(path, extensionList, listHidden=False):
    from os import listdir
    from os.path import isfile, join

    files = [f for f in listdir(path) if isfile(join(path, f))]
    rightFiles=[]
    for f in files:
        if listHidden==False:
            if f.startswith("."):
                continue

        for e in extensionList:
            if f.endswith(e):
                rightFiles.append(f)
                break
    return rightFiles

def _channels(color_mode = "rgb"):
    if color_mode == "rgb":
        channels = 3
    elif color_mode == "gray" or color_mode == "grayscale" or color_mode == "g" or color_mode == "gr":
        channels = 1
    return channels


def loadImageList(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch



def loadImageFolder(path, max_imgs=-1, img_size=None, crop_size=None,
                    color_mode="rgb", extensions=DEFAULT_EXTENSIONS, sort=False, out=None):
    imgFiles = _listOnlyFilesWithExt(path, extensions)
    if sort:
        imgFiles.sort()
    nImgs = max_imgs
    if max_imgs < 0 or max_imgs > len(imgFiles):
        nImgs = len(imgFiles)
        testSetSize = 0
        nImgs = len(imgFiles)

    if path[-1] != '/':
        path = path + '/'

    newImgFiles = []
    for imf in imgFiles:
        newImgFiles.append(path + imf)

    imgFiles = newImgFiles[0:nImgs]

    data = []
    if len(imgFiles) > 0:
        data = loadImageList(imgFiles, img_size, crop_size, color_mode, out)
    return data, imgFiles

def loadImagesWithLabel(path, label, max_imgs=-1, img_size=None, crop_size=None,
                               color_mode="rgb", extensions=DEFAULT_EXTENSIONS, out=None):
    data = loadImageFolder(path, max_imgs, img_size, crop_size, color_mode, extensions, out)
    labels = []
    for t in data:
        labels.append(label)
    return [data, labels]



def _unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def _unison_shuffled_copies_3(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def _unison_shuffled_copies_n(list_of_arrays):
    # type: (list(collections.Iterable)) -> list(collections.Iterable)

    not_none_map = {}
    old_len = -1

    # Ignore Nones in list_of_arrays
    for i, array in enumerate(list_of_arrays):
        if array is not None:
            not_none_map[i] = True
            assert (old_len == len(array) or old_len == -1)
            old_len = len(array)
        else:
            not_none_map[i] = False

    p = np.random.permutation(old_len)
    ret = []
    for i, array in enumerate(list_of_arrays):
        if not_none_map[i]:
            ret.append(array[p])
        else:
            ret.append(None)

    return ret

class ImageDataset:
    # load and manage an image dataset.
    # Can save to hdf5 file and load from a saved hdf5 file, numpy arrays, and folder containing folders.
    # Manage data, labels, label names (and file names).
    # Can save the content again in image files format.
    # TODO: not all methods are tested (in particular loadSingleImage and loadImagesFolder).
    # TODO: second label never implemented completely, not useful for now, for this reason has been commented.
    def __init__(self):
        self.__resetState()

    def __resetState(self):
        self.__setState(None, None, None, None)

    def __setState(self, dataset, labelset, filenames, labelmap=None):
        self.data = np.asarray(dataset) if dataset is not None else None
        self.labels = np.asarray(labelset) if labelset is not None else None   # labels[data_index]  = label_vector (numpy, [ 0, ... ,0, 1, 0, ... , 0]
        self.fnames = np.asarray(filenames) if filenames is not None else None  # labelmap[label_int] = label_name
        self.labelmap = np.asarray(labelmap) if labelmap is not None else None
        #self.second_labels = np.asarray(secondary_label) if secondary_label is not None else None


    def getLabelVec(self, data_index):
        if self.labels is not None:
            return self.labels[data_index]
        else: return None

    def getLabelInt(self, data_index):
        if self.labels is not None:
            return self.labelVecToInt(self.getLabelVec(data_index))
        else: return None

    def getLabelStr(self, data_index):
        return self.labelIntToStr(self.getLabelInt(data_index))

    # def getSecondLabelVec(self, data_index):
    #     if self.labels is not None:
    #         return self.second_labels[data_index]
    #     else: return None
    #
    # def getSecondLabelInt(self, data_index):
    #     if self.labels is not None:
    #         return self.labelVecToInt(self.getSecondLabelVec(data_index))
    #     else: return None
    #
    # def getSecondLabelStr(self, data_index):
    #     return self.labelIntToStr(self.getSecondLabelInt(data_index))



    def labelIntToStr(self, label_int):
        if ImageDataset.labelmap is not None:
            return ImageDataset.labelmap[label_int]
        else: return None

    def labelIntToVec(self, label_int):
        if self.labels is not None:
            v = np.zeros([len(self.labels)])
            v[:, label_int] = 1
            return v
        else: return None

    def labelVecToInt(self, label_vector):
        # np.argmax(label_vector, 0)
        return np.nonzero(label_vector)[0][0]

    def labelVecToStr(self, label_vector):
        return self.labelIntToStr(self.labelVecToInt(label_vector))

    def labelStrToInt(self, label_name_str):
        if ImageDataset.labelmap is not None:
            dict = {}
            for i, l in self.labelmap:
                dict[l] = i
            return dict[label_name_str]

    def labelStrToVec(self, label_name_str):
        index = self.labelStrToInt(label_name_str)
        if index is not None:
            return self.labelIntToVec(index)
        else: return None



    def getLabelsInt(self):
        return np.argmax(self.labels, axis=1)
        # label_list = []
        # for l in self.labels:
        #     label_list.append( self.labelVecToInt(l) )
        # return np.asarray(label_list)

    def getLabelsVec(self):
        return self.labels

    def getLabelsStr(self):
        label_list = []
        for l in self.labels:
            label_list.append(self.labelVecToStr(l))
        return np.asarray(label_list)

    # def getSecondLabelsInt(self):
    #     return np.argmax(self.second_labels, axis=1)
    #     # label_list = []
    #     # for l in self.labels:
    #     #     label_list.append( self.labelVecToInt(l) )
    #     # return np.asarray(label_list)
    #
    # def getSecondLabelsVec(self):
    #     return self.second_labels
    #
    # def getSecondLabelsStr(self):
    #     label_list = []
    #     for l in self.second_labels:
    #         label_list.append(self.labelVecToStr(l))
    #     return np.asarray(label_list)



    def shuffle(self):
        [self.data, self.labels, self.fnames] = _unison_shuffled_copies_n([self.data, self.labels, self.fnames])


    def loadSingleImage(self, img_path, labelname=None, labelvec=None,
                        crop_size=None, img_size=None, color_mode="rgb"):
        data = loadImageList([img_path], img_size, crop_size, color_mode)
        labels = None
        filenames= None
        label_names = None
        # second_labels = None

        nlabels = -1
        if labelvec is not None:
            labels = [labelvec]
            nlabels = len(labelvec)

        # if second_label_vec is not None:
        #     second_labels = [second_label_vec]
        #     if nlabels is -1:
        #         nlabels = len(second_label_vec)
        #     elif len(second_label_vec) is not nlabels:
        #         raise ValueError("Label and secondary label must be two vectors of the same dimension")

        if nlabels is not -1:
            label_names = []
            for i in range(0, nlabels):
                label_names.append("unknown_label_name_{}".format(i))

            if labelname is not None and nlabels is not -1:
                label_names[self.labelVecToInt(labelvec)] = labelname
            # if second_label_name is not None and nlabels is not -1:
            #     label_names[self.labelVecToInt(second_label_vec)] = second_label_name

        filenames = [ _path_leaf(img_path) ]
        self.__setState(data, labels, filenames, label_names)


    # TODO: AGGIUNGI QUA UN FLAG PER LEGGERE DOPPIE LABEL (datasetpath/noisy_label/true_label/imgs.jpg)
    def loadImagesFolder(self, path, max_imgs=-1, crop_size=None, img_size=None, color_mode="rgb",
                         imgExtensions=DEFAULT_EXTENSIONS, sortFileNames=False):
        from os.path import join

        [d, fname] = loadImageFolder(path=path, max_imgs=max_imgs,
                                     crop_size=crop_size, img_size=img_size,
                                     color_mode=color_mode, extensions=imgExtensions, sort=sortFileNames)

        dataset = np.zeros([0, _channels(color_mode), crop_size[0], crop_size[1]])
        dataset = np.concatenate([dataset, d], 0)
        for j in range(0,len(fname)):
            fname[j] = _path_leaf(fname[j])
        self.__setState(dataset, None, fname, None)

    def loadImagesDataset(self, dataset_path, max_img_per_label=-1,
                          crop_size=None, img_size=None, outlier_label = None,
                          color_mode="rgb", imgExtensions=DEFAULT_EXTENSIONS, sortFileNames=True, sortLabelNames=True):

        # Load dataset organized as: /dataset/label1/img1
        #
        # /dataset/label1/img1.jpg
        #                 img2.jpg
        #                 img3.jpg
        #                   ...
        #
        # /dataset/label2/img1.jpg
        #                 img2.jpg
        #                 img3.jpg
        #
        # ...
        # Setting outlier_label, this method will search for that label and set as last labe in the label ordering.

        from os.path import join
        labels = _listOnlyDirs(dataset_path)
        labelmap = []
        filenames = []
        dataset = np.zeros([0, 3, crop_size[0], crop_size[1]])  #[index,  channels, pixel_y, pixel_x] # TODO: infert crop_size[0] and 1?
        labelset = np.zeros([0, len(labels)])  # [index,   softmax_label]
        i=0
        labels.sort()

        if outlier_label is not None:
            outlier_label_found = False
            newlabels = []
            for l in labels:
                if l == outlier_label:
                    outlier_label_found = True
                else:
                    newlabels.append(l)
            if outlier_label_found:
                newlabels.append(outlier_label)

        for l in labels:
            [d, fname] = loadImageFolder(path=join(dataset_path, l), max_imgs=max_img_per_label,
                                         crop_size=crop_size, img_size=img_size,
                                         color_mode=color_mode, extensions=imgExtensions, sort=sortFileNames)
            #labelmap_str[l] = i
            labelmap.append(l)

            dataset = np.concatenate([dataset, d], 0)
            l = np.zeros([len(d), len(labels)])
            l[:, i] = 1
            labelset = np.concatenate([labelset, l], 0)
            for j in range(0,len(fname)): fname[j] = _path_leaf(fname[j])
            filenames = filenames + fname
            i+=1
        self.__setState(dataset, labelset, filenames, labelmap)

    def loadNumpyArray(self, data, labels=None, fnames=None, labelmap=None):
        #TODO: TEST
        # TODO: What about the maps?
        dim_d = dim_f = dim_l = -1
        if data is not None: dim_d = data.shape[0]
        if labels is not None: dim_l = labels.shape[0]
        if fnames is not None: dim_f = len(fnames)

        if labels is None and (labelmap is not None ):
            raise ValueError("labelmap can be used only if labels is not None")
        if labels is not None and labelmap is not None:
            if len(labelmap) != dim_l:
                ValueError("len(labelmap) must be equal to labels.shape[0]")


        if dim_d != -1 and dim_l != -1 and dim_l != -1:
            if not (dim_d == dim_f == dim_l):
                raise ValueError("data, labels and fnames must have the same length (shape[0])")
        elif dim_d != -1 and dim_l != -1:
            if dim_d != dim_l:
                raise ValueError("data and labels must have the same length (shape[0])")

        elif dim_d != -1 and dim_f != -1:
            if dim_d != dim_f:
                raise ValueError("data and fnames must have the same length (shape[0])")

        elif dim_l != -1 and dim_f != -1:
            if dim_l != dim_f:
                raise ValueError("labels and fnames must have the same length (shape[0])")

        elif dim_d == -1 and dim_l != -1 and dim_f != -1:
            raise ValueError("At east one of data, labels or fnames array must not be None")

        self.__setState(data, labels, fnames, labelmap)

    def loadHDF5(self, path):
        h5f = h5py.File(path, 'r')
        if DATA_DATASET_NAME not in h5f.keys() and LABELS_DATASET_NAME not in h5f.keys() and FILENAMES_DATASET_NAME not in h5f.keys():
            h5f.close()
            raise ValueError('Cannot find any dataset in this hdf5 file.')
        else:
            data = labels = fnames = labelsmap = None
            if DATA_DATASET_NAME in h5f.keys():
                data = h5f[ DATA_DATASET_NAME][:]
            if LABELS_DATASET_NAME in h5f.keys():
                labels = h5f[LABELS_DATASET_NAME][:]
            if FILENAMES_DATASET_NAME in h5f.keys():
                fnames = h5f[FILENAMES_DATASET_NAME][:]
            if LABELS_MAP_NAME in h5f.keys():
                labelsmap = h5f[LABELS_MAP_NAME][:]
            h5f.close()
            self.__setState(data, labels, fnames, labelsmap)


    def saveToHDF5(self, path, write_data=True, write_labels=True, write_fnames=True, write_labels_maps = True):
        h5f = h5py.File(path, 'w')
        if write_data and self.data is not None:
            h5f.create_dataset(DATA_DATASET_NAME, data=self.data)
        if write_labels and self.labels is not None:
            h5f.create_dataset(LABELS_DATASET_NAME,    data=self.labels)
        if write_fnames and self.fnames is not None:
            #asciiList = [n.decode("ascii", "ignore") for n in self.fnames]
            h5f.create_dataset(FILENAMES_DATASET_NAME, data=self.fnames)
        if write_labels_maps and self.labelmap is not None:
            h5f.create_dataset(LABELS_MAP_NAME, data=self.labelmap)

        h5f.close()

    def saveToImages(self, directory, folder_per_label = True, label_in_name = True, use_fnames_if_available = True):

        if self.data is None:
            if self.labels is None or folder_per_label == False:
                raise ValueError("Can't save to images: data is None, labels is None or folder_per_label disabled")

        use_fnames = False
        if self.fnames is not None and use_fnames_if_available:
            use_fnames = True


        if self.data is not None:

            if not exists(directory):
                makedirs(directory)

            d_index=0
            for d in self.data:
                filename=str(d_index)

                if label_in_name:
                    label_index = self.getLabelInt(d_index)
                    if label_index is not None:
                        filename += '_label_' + str(label_index)
                        label_name = self.labelIntToStr(label_index)
                        if label_name is not None: filename += '_' + label_name

                if use_fnames and self.fnames is not None:
                    filename += '_' + self.fnames[d_index]

                vismap_index=0
                for vismap in d:
                    # TODO: if use_fnames == True ...
                    scipy.misc.imsave(join(directory, (filename + '_featuremap_' + str(vismap_index) + '.jpg')), vismap)
                    vismap_index+=1
                d_index+=1


        elif folder_per_label and self.labels is not None:
            i = 0
            for l in self.labels:
                np.os.makedirs(np.os.path.join(directory, str(l))) # TODO: use the map instad str(l)  if available

                i+=1












































#
# def loadImagesTrainTestSet(path, trainSetSize=-1, testSetSize=0, label = None, img_size=None, crop_size=None, color_mode="rgb", extensions=DEFAULT_EXTENSIONS, out=None):
#     #from os import listdir
#     #imgFiles = listdir(path)
#     imgFiles = _listOnlyFilesWithExt(path, extensions)
#     imgFiles.sort()
#
#     if trainSetSize < 0 or trainSetSize > len(imgFiles):
#         trainSetSize = len(imgFiles)
#         testSetSize = 0
#         trainSetSize = len(imgFiles)
#     if testSetSize < 0 or trainSetSize + testSetSize > len(imgFiles):
#         testSetSize = len(imgFiles) - trainSetSize
#
#     if path[-1] != '/':
#         path = path + '/'
#
#     newImgFiles = []
#     for imf in imgFiles:
#         newImgFiles.append(path + imf)
#
#     imgFiles_train = newImgFiles[0:trainSetSize]
#     imgFiles_test = newImgFiles[trainSetSize:trainSetSize + testSetSize]
#
#     if len(imgFiles_train) > 0:
#         train = _loadImageList(imgFiles_train, img_size, crop_size, color_mode, out)
#     else: train = []
#     if len(imgFiles_test) > 0:
#         test = _loadImageList(imgFiles_test, img_size, crop_size, color_mode, out)
#     else: test = []
#
#     if label != None:
#         trainWL = {}
#         for t in train:
#             trainWL['data'] = t
#             trainWL['label'] = label
#
#         testWL = {}
#         for t in test:
#             testWL['data'] = t
#             testWL['label'] = label
#         train = trainWL
#         test = testWL
#
#     if testSetSize==0:
#         return train
#
#     else:
#         return [train, test]