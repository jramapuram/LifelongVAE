import numpy as np
import tensorflow.contrib.keras as K

from copy import deepcopy
from utils import zip_filter_unzip
from tensorflow.python.framework import dtypes


# An object that filters the classes of cifar10
class CIFAR_Class(object):
    def __init__(self, class_number, cifar10):
        self.input_size = len(cifar10.train.images[0])
        self.number = class_number  # the class to filter out
        self.blacklist = list(np.arange(11))  # remember: goes to 10
        self.blacklist.remove(self.number)
        self.classes = self.filter_classes(cifar10, self.blacklist)

    @staticmethod
    def filter_classes(cifar10, blacklist):
        classes = deepcopy(cifar10)
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
        raise Exception("no validation for CIFAR10")

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
                 normalize=True):
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

class CIFAR10:
    def __init__(self, one_hot):
        (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
        self.train = DataSet(x_train, y_train, one_hot)
        self.test = DataSet(x_test, y_test, one_hot)

        # XXX: for compatibility
        self.number = 99999

    def get_batch_iter(self, batch_size):
        images, labels = self.train.next_batch(batch_size)
        return np.array(images), np.array(labels)

# Read mnist only once [~ 230Mb]
cifar10 = CIFAR10(one_hot=False)
