#
# Created by Amirreza Saghafian on 2/11/18.
#


class ClassificationInterface(object):
    """Classification interface.
    Using strategy design pattern.
    This interface will be used to define a family of
    classification algorithms.
    """
    def learn(self, features, labels):
        """Train the model.
        """
        pass

    def set_hyperparameters(self, parameters):
        """Set hyperparameters.

        Args:
            parameters: dict
                dictionary containing model hyper-parameters
        """
        pass

    def predict(self, x):
        """Use the trained model to predict labels.

        Args:
            x: ndarray
                array of features.

        Return:
            y: ndarray
                labels corresponding to x
        """
        pass
