#
# Created by Amirreza Saghafian on 2/11/18.
#


class LoadDataInterface(object):
    """Load data interface.
    Use strategy design pattern.
    Concrete strategies will implement this interface.
    """
    def load_data(self):
        pass

    def get_features_training(self):
        pass

    def get_labels_training(self):
        pass

    def get_features_test(self):
        pass

    def get_labels_test(self):
        pass

    def plot(self):
        pass
