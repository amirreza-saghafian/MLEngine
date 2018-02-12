#
# Created by Amirreza Saghafian on 2/11/18.
#

from src.classification.classification_interface import ClassificationInterface

from sklearn import svm
from sklearn import metrics
import numpy as np


class ClassificationSvmStrategy(ClassificationInterface):
    """Support vector machine classification strategy.
    Use sklearn svm classifier.
    Class is designed using strategy pattern.
    """
    def __init__(self, verbose=False):
        self.__classifier = svm.SVC(verbose=verbose)

    def set_hyperparameters(self, parameters):
        for key, value in parameters.items():
            if key == 'gamma':
                self.__classifier.set_params(gamma=parameters['gamma'])
            else:
                raise NotImplementedError('parameter %s has not set yet!'
                                          % key)

    def learn(self, features, labels):
        """Learn model parameters using sklearn svm.

        Args:
            features: ndarray
                array of training set features.
            labels: ndarray
                array of training set labels.
        """
        self.__classifier.fit(features, labels)
        # print("Classification report for classifier %s:\n%s\n"
        #       % (
        #       self.__classifier, metrics.classification_report(expected, predicted)))

    def predict(self, x, y_expected=None, show_misclassified_examples=False):
        """Predict labels for features x.

        Args:
            x: ndarray
                array of features.
            y_expected: ndarray, default=None
                array of expected labels (ground truth!)
            show_misclassified_examples: bool
                if true prints misclassified examples

        Returns:
            y_predicted: ndarray
                array of predicted labels
        """
        y_predicted = self.__classifier.predict(x)

        # computes training error if y_expected is provided
        if y_expected is not None:
            classification_error = float(np.sum(y_predicted != y_expected)) / \
                                   len(y_expected)

            print("error = %2.2f %%\n %s:\n%s\n"
                  % (classification_error * 100, self.__classifier,
                     metrics.classification_report(y_expected, y_predicted)))

            if show_misclassified_examples:
                for i, y in enumerate(y_predicted):
                    if y != y_expected[i]:
                        print('miss classified example %d. '
                              '\ny_expected = %d; y_predicted: %d\n'
                              % (i, y_expected[i], y))
