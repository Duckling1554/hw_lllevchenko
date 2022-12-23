from connector.pg_connector import get_data
from conf.conf import settings
from conf.conf import logging
from model.logistic_regression import train_logistic_regression
from model.svm import train_svm
from model.model_funcs import predict, split

TRAIN_DICT = {'SVM': train_svm,
              'LOG_REG': train_logistic_regression}

def initialize_model(model_name: str) -> None:
    """
    Function used for first-run of the model to train it on data stored in [DATA] in conf/settings.toml
    and hupertune with GridSearch.
    :param model_name: name of the model (taken from params in section [MODEL] in conf/settings.toml)
    """

    logging.info(f'Initializing model {model_name}')

    # Preparing train/test datasets
    df = get_data(settings.DATA.DATASET)

    X_train, X_test, y_train, y_test = split(df)

    # Initializing the model
    clf = TRAIN_DICT[model_name](X_train, y_train)

    logging.info(f'Accuracy is {clf.score(X_test,y_test)}')

    logging.info(f'Prediction is {predict(X_test, settings.MODEL.SVM)}')

    logging.info(f'Model {model_name} is ready for work!')