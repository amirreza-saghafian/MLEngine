#
# Created by Amirreza Saghafian on 2/11/18.
#

from src.model.classification_model import ClassificationModel
from src.load_data.load_data_sklearn_strategy \
    import LoadDataSklearnStrategy
from src.classification.classification_svm_strategy \
    import ClassificationSvmStrategy
from src.load_data.load_data_tensorflow_strategy \
    import LoadDataTensorflowStrategy
from src.classification.classification_cnn_strategy \
    import ClassificationCnnStrategy
from src.classification.classification_dnn_strategy \
    import ClassificationDnnStrategy


def main():
    print('Dataset options are: 1) MNIST from TensorFlow, '
          '2) DIGITS from scikit-learn')
    dataset = input('type dataset ID in terminal (DIGITS, or MNIST)? ')

    print('\nClassification options are: 1) SVM from scikit-learn, '
          '2) Deep Neural Network from TensorFlow.')
    classifier = input('type classifier ID in terminal (SVM or DNN)? ')

    print('\nyou have chosen: %s and %s.' % (dataset, classifier))
    print('Note: if you want to change model parameters check main.py file.\n')

    if dataset == 'DIGITS':
        load_data_obj = LoadDataSklearnStrategy(dataset, test_set_fraction=0.3)
    elif dataset == 'MNIST':
        load_data_obj = LoadDataTensorflowStrategy('mnist')
    else:
        raise ValueError('\n\n%s is not an option for dataset.'
                         '\nOptions are: DIGITS, or MNIST\n' % dataset)

    if classifier == 'SVM':
        classifier_obj = ClassificationSvmStrategy(verbose=False)
        gamma = 0.001
        classifier_obj.set_hyperparameters({'gamma': gamma})
    elif classifier == 'DNN':
        classifier_obj = ClassificationDnnStrategy()
        hyperparameters = {'hidden_units': [75, 50, 25], # size of hidden layers
                           'learning_rate': 1.0e-4,
                           'dropout': 0.1,
                           'batch_size': 128,
                           'num_steps': 50000,
                           'num_epochs': 1000}
        classifier_obj.set_hyperparameters(hyperparameters)
    else:
        raise ValueError('\n\n%s is not an option for classification.'
                         '\nOptions are: SVM or DNN\n' % classifier)

    # using strategy design patten can easily build the custom model
    model = ClassificationModel(load_data_obj, classifier_obj)
    model.learn()  # train the model
    model.predict()  # evaluate the model on the test set


if __name__ == '__main__':
    main()
