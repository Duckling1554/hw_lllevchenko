from conf.conf import logging
from sklearn.linear_model import LogisticRegression
import pandas as pd
from util.util import save_model
from conf.conf import settings


def train_logistic_regression(X_train:pd.DataFrame, y_train:pd.DataFrame) -> LogisticRegression:
    """
    Function used to train LOG REG (logistic regression) and save model config to pkl file.
    :param X_train: train df of parameters
    :param y_train: train df of target
    :return: LOG REG model
    """

    # Initialize the model
    clf = LogisticRegression()

    logging.info("Training the model")
    # Train the model
    clf.fit(X_train, y_train)

    save_model(dir=settings.MODEL.LOG_REG, model=clf)

    return clf
