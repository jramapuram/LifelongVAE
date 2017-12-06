import os
import cv2
import numpy as np
import tensorflow.contrib.keras as K
from scipy.misc import imrotate as rotate
from scipy.misc import imresize as imresize
from sklearn.preprocessing import StandardScaler


from copy import deepcopy
from utils import zip_filter_unzip
from tensorflow.python.framework import dtypes


TRAIN_IMGS_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
TRAIN_LABLES_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
TEST_IMAGES_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
TEST_LABELS_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
TRAIN_IMGS_FILE = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS_FILE ='train-labels-idx1-ubyte.gz'
TEST_IMGS_FILE = 't10k-images-idx3-ubyte.gz'
TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz'


# An object that filters the classes of fashion
class Fashion_Class(object):
    def __init__(self, class_number, fashion,
                 is_flat=True, resize_dims=None,
                 convert_to_rgb=False):
        self.input_size = len(fashion.train.images[0])
        self.number = class_number  # the class to filter out
        self.blacklist = list(np.arange(11))  # remember: goes to 10
        self.blacklist.remove(self.number)
        self.classes = self.filter_classes(fashion, self.blacklist)

        # return images in [batch, row, col]
        if not is_flat:
            self.classes = Fashion_Class._unflatten_mnist(self.classes)

        # resizes images if resize_dims tuple is provided
        if resize_dims is not None:
            self.classes = Fashion_Class.resize_mnist(self.classes, resize_dims)

        # tile images as [img, img, img]
        if convert_to_rgb:
            self.classes = Fashion_Class.bw_to_rgb_mnist(self.classes)

    @staticmethod
    def _unflatten_mnist(mnist):
        mnist.train._images = mnist.train._images.reshape([-1, 28, 28])
        mnist.test._images = mnist.test._images.reshape([-1, 28, 28])
        return mnist

    @staticmethod
    def resize_mnist(mnist, new_dims):
        mnist.train._images = Fashion_Class.resize_images(mnist.train._images, new_dims)
        mnist.test._images = Fashion_Class.resize_images(mnist.test._images, new_dims)
        return mnist

    @staticmethod
    def bw_to_rgb_mnist(mnist):
        mnist.train._images = Fashion_Class.bw_to_rgb(mnist.train._images)
        mnist.test._images = Fashion_Class.bw_to_rgb(mnist.test._images)
        return mnist


    @staticmethod
    def resize_images(imgs, new_dims):
        tuple_size = tuple(new_dims) if type(new_dims) == list else new_dims
        return np.vstack([np.expand_dims(cv2.resize(img.reshape(28, 28),
                                                    tuple_size), 0)
                          for img in imgs])

        # return np.vstack([imresize(img.reshape(28, 28),
        #                            new_dims, mode='L',
        #                            interp='lanczos').reshape(flattened_dims)
        #                   for img in imgs])

    @staticmethod
    def bw_to_rgb(imgs):
        return np.vstack([np.tile(img.reshape(img.shape[0], imgs.shape[1], 1), 3)
                          .reshape(-1, img.shape[0], img.shape[1], 3)
                          for img in imgs])

    @staticmethod
    def _rotate_batch(batch, angle):
        return np.vstack([rotate(x_i.reshape(28, 28), angle).reshape([-1, 28*28])
                          for x_i in batch])

    @staticmethod
    def _check_and_load_angle(angle, number, base_path='MNIST_data'):
        ''' Returns None if the file doesn't exists'''
        filename = os.path.join(base_path, "mnist_num%d_angle%d.hdf5"
                                % (number, angle))
        if os.path.exists(filename):
            f = h5py.File(filename, "r")
            return f['train'][()], f['test'][()]
            # return f['train'], f['validation'], f['test']

        return None

    @staticmethod
    def _check_and_write_angle(angle, number, mnist, base_path='MNIST_data'):
        ''' serializes the rotated number to disk as a hdf5 file'''
        filename = os.path.join(base_path, "mnist_num%d_angle%d.hdf5"
                                % (number, angle))
        if not os.path.exists(filename):
            f = h5py.File(filename, "w")
            f['train'] = mnist.train._images
            f['test'] = mnist.test._images

            print 'serialized %s to disk...' % filename

    @staticmethod
    def rotate_all_sets(mnist, number, angle):
        hpf5_load = Fashion_Class._check_and_load_angle(angle, number)
        if hpf5_load is not None:
            train_imgs = np.asarray(hpf5_load[0], np.float32)
            test_imgs = np.asarray(hpf5_load[2], np.float32)
        else:
            train_imgs = Fashion_Class._rotate_batch(mnist.train._images, angle)
            test_imgs = Fashion_Class._rotate_batch(mnist.test._images, angle)

        mnist.train._images = train_imgs
        mnist.test._images = test_imgs

        Fashion_Class._check_and_write_angle(angle, number, mnist)
        return mnist

    @staticmethod
    def filter_classes(fashion, blacklist):
        classes = deepcopy(fashion)
        classes.train._images, classes.train._labels = zip_filter_unzip(classes.train._images,
                                                                        classes.train._labels,
                                                                        blacklist)
        classes.train._images = np.array(classes.train._images)
        classes.train._labels = np.array(classes.train._labels)
        classes.train._num_examples = len(classes.train.images)
        classes.test._images, classes.test._labels = zip_filter_unzip(classes.test._images,
                                                                      classes.test._labels,
                                                                      blacklist)
        classes.test._images = np.array(classes.test._images)
        classes.test._labels = np.array(classes.test._labels)
        classes.test._num_examples = len(classes.test.images)
        return classes

    # if one vs. all then 0 = true class, 1 = other
    # otherwise we just use lbl = lbl,  10 = other
    def _augment(self, images, labels):
        indexer = np.array(labels == self.number)
        return zip(*((im, lbl) if ind else (im, 10)
                     for im, lbl, ind in zip(images, labels, indexer)))

    def get_train_batch_iter(self, batch_size):
        images, labels = self.classes.train.next_batch(batch_size)
        #images, labels = self._augment(images, labels)
        return np.array(images), np.array(labels)

    def get_validation_batch_iter(self, batch_size):
        raise Exception("no validation for Fashion")

    def _get_test_batch_iter(self, batch_size):
        images, labels = self.classes.test.next_batch(batch_size)
        images, labels = self._augment(images, labels)
        return np.array(images), np.array(labels)

    def get_test_batch_iter(self, batch_size):
        images = []; labels = []; count = 0
        while(count < batch_size):
            max_batch = self.classes.test._num_examples
            im, lbl = self._get_test_batch_iter(max_batch)
            tar = self.number
            if tar in lbl:
                im, lbl = zip_filter_unzip(im, lbl, self.blacklist)
                count += len(im)
                #  im = np.asarray(im); lbl = np.asarray(lbl); count += len(lbl)
                images.append(im); labels.append(lbl)

        return np.vstack(images)[0:batch_size], np.hstack(labels)[0:batch_size]

    def get_batch_iter(self, batch_size):
        images = []; labels = []; count = 0
        while(count < batch_size):
            im, lbl = self.get_train_batch_iter(batch_size)
            tar = self.number
            if tar in lbl:
                # im, lbl = zip_filter_unzip(im, lbl, self.blacklist)
                im = np.asarray(im); lbl = np.asarray(lbl); count += len(lbl)
                images.append(im); labels.append(lbl)

        return np.vstack(images)[0:batch_size], np.hstack(labels)[0:batch_size]


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32,
                 normalize=False):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            if normalize:
                images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return [np.concatenate((images_rest_part, images_new_part), axis=0),
                    np.concatenate((labels_rest_part, labels_new_part), axis=0)]
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

class Fashion:
    def __init__(self, one_hot, path='Fashion_data'):
        # (x_train, y_train), (x_test, y_test) = K.datasets.fashion.load_data()
        self.download(path)
        x_train, y_train = self.load_mnist(path, kind='train')
        # x_train = x_train.reshape(-1, 28, 28)
        x_test, y_test = self.load_mnist(path, kind='t10k')
        # x_test = x_test.reshape(-1, 28, 28)

        self.train = DataSet(x_train, y_train, one_hot)
        self.test = DataSet(x_test, y_test, one_hot)

        # XXX: for compatibility
        self.number = 9996

    def get_batch_iter(self, batch_size):
        images, labels = self.train.next_batch(batch_size)
        return np.array(images), np.array(labels)

    @staticmethod
    def normalize_imgs(imgs_train, imgs_test):
        imgs_train_scaled = imgs_train / 255.
        imgs_test_scaled = imgs_test / 255.

        s = StandardScaler()
        s = s.fit(imgs_train_scaled)
        imgs_train = s.transform(imgs_train_scaled)
        imgs_test = s.transform(imgs_test_scaled)
        return imgs_train_scaled, imgs_test_scaled

    @staticmethod
    def load_mnist(path, kind='train'):
        ''' From https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/utils/mnist_reader.Pu '''
        import os
        import struct
        import gzip
        import numpy as np

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            struct.unpack('>II', lbpath.read(8))
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            struct.unpack(">IIII", imgpath.read(16))
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    @staticmethod
    def _exists(path):
        train_imgs_path, train_labels_path, \
            test_imgs_path, test_labels_path = Fashion.get_paths(path)
        return os.path.isdir(path) and os.path.exists(train_imgs_path) \
            and os.path.exists(train_labels_path) \
            and os.path.exists(test_imgs_path) \
            and os.path.exists(test_labels_path)

    @staticmethod
    def get_paths(path):
        train_imgs_path = os.path.join(path, TRAIN_IMGS_FILE)
        train_labels_path = os.path.join(path, TRAIN_LABELS_FILE)
        test_imgs_path = os.path.join(path, TEST_IMGS_FILE)
        test_labels_path = os.path.join(path, TEST_LABELS_FILE)
        return [train_imgs_path, train_labels_path,
                test_imgs_path, test_labels_path]

    def download(self, path):
        '''Note: path is the base dir '''
        if not self._exists(path):
            if not os.path.isdir(path):
                os.makedirs(path)

            # gather file paths
            zip_files = Fashion.get_paths(path)
            # zip_files = [TRAIN_IMGS_FILE, TRAIN_LABELS_FILE,
            #              TEST_IMGS_FILE, TEST_LABELS_FILE]
            # zip_files = [os.path.join(path, z) for z in zip_files]

            # gather urls
            urls = [TRAIN_IMGS_URL, TRAIN_LABLES_URL,
                    TEST_IMAGES_URL, TEST_LABELS_URL]

            # download the file(s)
            import urllib
            for filename, url in zip(zip_files,  urls):
                print("downloading ", filename)
                urllib.urlretrieve(url=url, filename=filename)

            print("FashionMNIST downloaded successfully...")
        else:
            print("FashionMNIST files already downloaded...")


def normalize_images(imgs, mu=None, sigma=None, eps=1e-9):
    ''' normalize imgs with provided mu /sigma
        or computes them and returns with the normalized
       images '''
    if mu is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[-1]
            mu = np.asarray(
                [np.mean(imgs[:, :, :, i]) for i in range(chans)]
            ).reshape(1, 1, 1, -1)
        else:
            raise Exception("unknown number of dims for normalization")

    if sigma is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[-1]
            sigma = np.asarray(
                [np.std(imgs[:, :, :, i]) for i in range(chans)]
            ).reshape(1, 1, 1, -1)
        else:
            raise Exception("unknown number of dims for normalization")

    return (imgs - mu) / (sigma + eps), [mu, sigma]

def normalize_train_test_images(train_imgs, test_imgs, eps=1e-9):
    ''' simple helper to take train and test images
        and normalize the test images by the train mu/sigma '''
    assert len(train_imgs.shape) == len(test_imgs.shape) >= 4

    train_imgs , [mu, sigma] = normalize_images(train_imgs, eps=eps)
    return [train_imgs,
            (test_imgs - mu) / (sigma + eps)]

def scale(val, src, dst):
    """Helper to scale val from src range to dst range
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


# Read mnist only once [~ 230Mb]
fashion = Fashion(one_hot=False)

# Dense
fashion.train._images = fashion.train._images.reshape([-1, 28, 28]).astype(np.uint8)
fashion.test._images = fashion.test._images.reshape([-1, 28, 28]).astype(np.uint8)

################ Method: OTSU #############
# train = []
# for img in fashion.train._images:
#     _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     train.append(th)

# fashion.train._images = np.vstack(train)

# test = []
# for img in fashion.test._images:
#     _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     test.append(th)

# fashion.test._images = np.vstack(test)
###########################################

######### Method: adaptive thresholding ###############
fashion.train._images = np.vstack([cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                         cv2.THRESH_BINARY, 21, 0) for img in fashion.train._images])
fashion.test._images = np.vstack([cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                        cv2.THRESH_BINARY, 21, 0) for img in fashion.test._images])

fashion.train._images = fashion.train._images.reshape([-1, 28*28]).astype(np.float32)
fashion.test._images = fashion.test._images.reshape([-1, 28*28]).astype(np.float32)
fashion.train._images /= 255.0
fashion.test._images /= 255.0
print("POST")
print("fashion train min = ", np.min(fashion.train._images))
print("fashion train max = ", np.max(fashion.train._images))
print("fashion test min = ", np.min(fashion.test._images))
print("fashion test max = ", np.max(fashion.test._images))
print("fashion shape = ", fashion.train._images.shape)
#########################################################


# CONV
# fashion.train._images = fashion.train._images.reshape([-1, 28, 28]).astype(np.uint8)
# fashion.test._images = fashion.test._images.reshape([-1, 28, 28]).astype(np.uint8)
# fashion.train._images = Fashion_Class.resize_images(fashion.train._images, [32, 32])
# fashion.test._images = Fashion_Class.resize_images(fashion.test._images, [32, 32])
# print("PRE")
# print('fs imgs = ', fashion.train._images.shape)
# print("fashion train min = ", np.min(fashion.train._images))
# print("fashion train max = ", np.max(fashion.train._images))
# print("fashion test min = ", np.min(fashion.test._images))
# print("fashion test max = ", np.max(fashion.test._images))
# print("dtype = ", fashion.train._images.dtype)

# ######### Method: adaptive thresholding ###############
# fashion.train._images = np.vstack([np.expand_dims(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                                                                         cv2.THRESH_BINARY, 21, 0), 0)
#                                    for img in fashion.train._images]).astype(np.float32)
# fashion.test._images = np.vstack([np.expand_dims(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                                                                        cv2.THRESH_BINARY, 21, 0), 0)
#                                   for img in fashion.test._images]).astype(np.float32)
# print("train shp = ", fashion.train._images.shape)
# fashion.train._images /= 255.0
# fashion.test._images /= 255.0
# fashion.train._images = Fashion_Class.bw_to_rgb(fashion.train._images)
# fashion.test._images = Fashion_Class.bw_to_rgb(fashion.test._images)
# print("POST")
# print("fashion train min = ", np.min(fashion.train._images))
# print("fashion train max = ", np.max(fashion.train._images))
# print("fashion test min = ", np.min(fashion.test._images))
# print("fashion test max = ", np.max(fashion.test._images))
# print("fashion shape = ", fashion.train._images.shape)
# #########################################################

# # fashion.train._images, fashion.test._images \
# #     = normalize_train_test_images(fashion.train._images, fashion.test._images)

# # fashion.train._images = scale(fashion.train._images, [np.min(fashion.train._images),
# #                                                       np.max(fashion.train._images)], [0.0, 1.0])
# # fashion.test._images = scale(fashion.test._images, [np.min(fashion.test._images),
# #                                                       np.max(fashion.test._images)], [0.0, 1.0])

# # print("fashion train min = ", np.min(fashion.train._images))
# # print("fashion train max = ", np.max(fashion.train._images))
# # print("fashion test min = ", np.min(fashion.test._images))
# # print("fashion test max = ", np.max(fashion.test._images))

# # # fashion.train._images = fashion.train._images.reshape([-1, int(32*32*3)])
# # # fashion.test._images = fashion.test._images.reshape([-1, int(32*32*3)])
# # # fashion.train._images, fashion.test._images \
# # #     = Fashion.normalize_imgs(fashion.train._images, fashion.test._images)
# # # fashion.train._images = fashion.train._images.reshape([-1, 32, 32, 3])
# # # fashion.test._images = fashion.test._images.reshape([-1, 32, 32, 3])
