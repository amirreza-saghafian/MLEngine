#
# Created by Amirreza Saghafian on 2/11/18.
#


class ClassificationModel(object):
    """Loads data and assigns training, validation, and test sets.
    Using strategy design pattern, this class takes a concrete load_data
    strategy and a concrete classification strategy.
    """
    def __init__(self, load_data_concrete, classifier_concrete):
        self.__load_data_obj = load_data_concrete
        self.__classifier_obj = classifier_concrete
        self.__x_train, self.__y_train = self.__load_training_data()
        self.__x_test, self.__y_test = self.__load_test_data()
        print('shape of training features: ', self.__x_train.shape)
        print('shape of training labels: ', self.__y_train.shape)

        if self.__x_test is not None:
            print('shape of test features: ', self.__x_test.shape)
            print('shape of test labels: ', self.__y_test.shape)

    def __load_training_data(self):
        self.__load_data_obj.load_data()
        return self.__load_data_obj.get_features_training(), \
               self.__load_data_obj.get_labels_training()

    def __load_test_data(self):
        return self.__load_data_obj.get_features_test(), \
               self.__load_data_obj.get_labels_test()

    def learn(self):
        self.__classifier_obj.learn(self.__x_train, self.__y_train)
        print('\nTraining Error:')
        print('-----------------------------')
        self.__classifier_obj.predict(self.__x_train, self.__y_train)

    def predict(self):
        print('\nTest Error:')
        print('-----------------------------')
        self.__classifier_obj.predict(self.__x_test, self.__y_test)
