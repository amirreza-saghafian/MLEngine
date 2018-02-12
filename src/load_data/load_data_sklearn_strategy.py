#
# Created by Amirreza Saghafian on 2/11/18.
#

from src.load_data.load_data_interface import LoadDataInterface

import numpy as np
from sklearn import datasets


class LoadDataSklearnStrategy(LoadDataInterface):
    """Load data from sklearn datasets.
    It is a concrete strategy using LoadDataInterface.
    """
    def __init__(self, dataset_name, test_set_fraction=None):
        """Constructor of class.

        Args:
            dataset_name: string
                name of the dataset to load
            test_set_fraction: float, default=0
                size of test set fraction
        """
        self.__dataset_id = dataset_name
        self.__data = None
        self.__num_training_examples = None
        self.__num_test_examples = None

        if test_set_fraction is None:
            self.__test_set_fraction = 0.0
        else:
            self.__test_set_fraction = test_set_fraction

    def load_data(self):
        if self.__dataset_id == 'DIGITS':
            self.__data = datasets.load_digits()
            self.__num_training_examples = int(len(self.__data.images) *
                                               (1.0 - self.__test_set_fraction))
            self.__num_test_examples = len(self.__data.images) - \
                                       self.__num_training_examples
        elif self.__dataset_id == 'IRIS':
            self.__data = datasets.load_iris()
            print('iris: ', self.__data['data'].shape)
            self.__num_training_examples = int(len(self.__data['data']) *
                                               (1.0 - self.__test_set_fraction))
            self.__num_test_examples = len(self.__data['data']) - \
                                       self.__num_training_examples
        else:
            raise NotImplementedError('Cannot load %s yet!'
                                      % self.__dataset_id)

    def get_features_training(self):
        if self.__dataset_id == 'DIGITS':
            return np.array(
                self.__data.images[:self.__num_training_examples].reshape(
                    (self.__num_training_examples, -1)), dtype=np.float32
            )
        elif self.__dataset_id == 'IRIS':
            return np.array(
                self.__data['data'][:self.__num_training_examples].reshape(
                    (self.__num_training_examples, -1)), dtype=np.float32
            )

    def get_labels_training(self):
        if self.__dataset_id == 'DIGITS':
            return np.array(self.__data.target[:self.__num_training_examples],
                            dtype=np.int32)
        elif self.__dataset_id == 'IRIS':
            return np.array(self.__data['target']
                            [:self.__num_training_examples],
                            dtype=np.int32)


    def get_features_test(self):
        if self.__test_set_fraction > 0:
            if self.__dataset_id == 'DIGITS':
                return np.array(
                    self.__data.images[self.__num_training_examples:].reshape(
                        (self.__num_test_examples, -1)), dtype=np.float32
                )
            elif self.__dataset_id == 'IRIS':
                return np.array(
                    self.__data['data'][self.__num_training_examples:].reshape(
                        (self.__num_test_examples, -1)), dtype=np.float32
                )

        else:
            return None

    def get_labels_test(self):
        if self.__test_set_fraction > 0:
            if self.__dataset_id == 'DIGITS':
                return np.array(
                    self.__data.target[self.__num_training_examples:],
                    dtype=np.int32)
            elif self.__dataset_id == 'IRIS':
                return np.array(
                    self.__data['target'][self.__num_training_examples:],
                    dtype=np.int32)

        else:
            return None
