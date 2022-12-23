from connector.pg_connector import get_data
from conf.conf import settings
from model.split import split
from model.logistic_regression import train_logistic_regression
from model.svm import train_svm

from model.logistic_regression import predict
from conf.conf import logging


df = get_data(settings.DATA.DATASET)
X_train, X_test, y_train, y_test = split(df)
clf = train_svm(X_train, y_train)

logging.info(f'Accuracy is {clf.score(X_test,y_test)}')

logging.info(f'Prediction is {predict(X_test, settings.MODEL.PATH_SVM)}')