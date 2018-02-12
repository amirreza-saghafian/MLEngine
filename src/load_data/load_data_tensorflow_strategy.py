#
# Created by Amirreza Saghafian on 2/11/18.
#

from src.load_data.load_data_interface import LoadDataInterface

import numpy as np
import tensorflow as tf


class LoadDataTensorflowStrategy(LoadDataInterface):
    """Load data from tensorflow database.
    Implements interface class LoadDataInterface based on
    strategy design pattern.
    """

    def __init__(self, database_name, num_training_examples=None,
                 num_test_examples=None):
        self.__dataset_id = database_name
        self.__features_training = None
        self.__labels_training = None
        self.__features_test = None
        self.__labels_test = None
        self.__num_training_examples = num_training_examples
        self.__num_test_examples = num_test_examples

    def load_data(self):
        if self.__dataset_id == 'mnist':
            mnist = tf.contrib.learn.datasets.load_dataset('mnist')

            if self.__num_training_examples is None:
                self.__num_training_examples = len(mnist.train.images)
            elif self.__num_training_examples > len(mnist.train.images):
                raise ValueError('Number of training examples to load is '
                                 'larger than dataset!')

            self.__features_training = \
                mnist.train.images[:self.__num_training_examples]
            self.__labels_training = np.asarray(
                mnist.train.labels[:self.__num_training_examples],
                dtype=np.int32)

            if self.__num_test_examples is None:
                self.__num_test_examples = len(mnist.test.images)
            elif self.__num_test_examples > len(mnist.test.images):
                raise ValueError('Number of test examples to load is '
                                 'larger than dataset!')

            self.__features_test = mnist.test.images[:self.__num_test_examples]
            self.__labels_test = np.asarray(
                mnist.test.labels[:self.__num_test_examples], dtype=np.int32)
        else:
            raise NotImplementedError('Cannot load %s yet! sorry!'
                                      % self.__dataset_id)

    def get_features_training(self):
        return self.__features_training

    def get_labels_training(self):
        return self.__labels_training

    def get_features_test(self):
        return self.__features_test

    def get_labels_test(self):
        return self.__labels_test
