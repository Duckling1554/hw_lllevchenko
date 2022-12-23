from conf.conf import settings
from util.util import load_model
from conf.conf import logging

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV



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


def grid_search(model_name: str, X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    """
    Hypertuning using GridSearch.
    :param model_name: name of the model (from [MODEL] in conf/settings.toml)
    :param X_train: train df of parameters
    :param y_train: train df of target
    :return: dict of best params
    """

    model = load_model(settings.MODEL[model_name])

    logging.info('Starting GridSearch')
    params = settings.MODEL[model_name+'_GRID']
    searcher = GridSearchCV(model, params)
    searcher.fit(X_train, y_train)
    logging.info(f'Best parameters: {searcher.best_params_}')
    return searcher.best_params_
