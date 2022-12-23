from connector.pg_connector import get_data
from conf.conf import settings
from model.logistic_regression import train_logistic_regression
from model.svm import train_svm
from util.util import load_model
from conf.conf import logging
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_DICT = {'SVM': train_svm,
              'LOG_REG': train_logistic_regression}


def split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split input df to X_train, X_test, y_train, y_tes
    :param df: input dataframe to be split
    :return: X_train, X_test, y_train, y_tes
    """

    logging.info("Defining X and y")

    # Filter out target column and take all other columns
    X = df.iloc[:, :-1]
    # Select target column
    y = df['target']

    logging.info("Splitting dataset")
    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,  # independent variables
                                                        y  # dependent variable
                                                        )
    return X_train, X_test, y_train, y_test


def predict(values: list, path_to_model: str) -> list:
    """
    Prediction using one of pretrained models.
    :param values: input vector for prediction
    :param path_to_model: path to pkl file with model config (from conf/settings.toml)
    :return: prediction vector
    """

    clf = load_model(path_to_model)

    logging.info('Starting predicting')
    return clf.predict(values)


def initialize_model(model_name: str) -> None:
    """
    Function used for first-run of the model to train it on data stored in [DATA] in conf/settings.toml.
    :param model_name: name of the model (taken from params in section [MODEL] in conf/settings.toml)
    """

    logging.info(f'Initializing model {model_name}')

    # Preparing train/test datasets
    df = get_data(settings.DATA.DATASET)

    X_train, X_test, y_train, y_test = split(df)

    # Training the model
    clf = TRAIN_DICT[model_name](X_train, y_train)

    logging.info(f'Accuracy is {clf.score(X_test,y_test)}')

    logging.info(f'Prediction is {predict(X_test, settings.MODEL.SVM)}')

    logging.info(f'Model {model_name} is ready for work!')
