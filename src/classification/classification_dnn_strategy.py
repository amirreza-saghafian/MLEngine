#
# Created by Amirreza Saghafian on 2/11/18.
#

from src.classification.classification_interface import ClassificationInterface

import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


class ClassificationDnnStrategy(ClassificationInterface):
    """Convolutional neural network classifier using TensorFlow.
    This strategy implements the interface class ClassificationInterface.
    """
    def __init__(self):
        self.__classifier = None
        self.__hyperparameters = None

    def learn(self, features, labels):
        # print('labels shape: ', labels.shape)
        num_classes = len(np.unique(labels))
        print('number of classes = %d ' % num_classes)

        feature_columns = [
            tf.feature_column.numeric_column("x", shape=features.shape[1])]

        self.__classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=self.__hyperparameters['hidden_units'],
            optimizer=tf.train.AdamOptimizer(
                self.__hyperparameters['learning_rate']),
            n_classes=num_classes,
            # model_dir="./tmp/mnist_model"
            dropout=self.__hyperparameters['dropout']
        )

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': features},
            y=labels,
            num_epochs=self.__hyperparameters['num_epochs'],
            batch_size=self.__hyperparameters['batch_size'],
            shuffle=True
        )
        self.__classifier.train(input_fn=train_input_fn,
                                steps=self.__hyperparameters['num_steps'])

    def set_hyperparameters(self, parameters):
        self.__hyperparameters = parameters

    def predict(self, x, y_expected=None):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x},
            y=y_expected,
            num_epochs=1,
            shuffle=False
        )

        accuracy_score = self.__classifier.evaluate(
            input_fn=test_input_fn)["accuracy"]
        print("\nAccuracy: {0:f}%\n".format(accuracy_score * 100))
