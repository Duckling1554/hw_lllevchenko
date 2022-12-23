from conf.conf import logging
from sklearn.svm import SVC
import pandas as pd
from util.util import save_model
from conf.conf import settings


def train_svm(X_train:pd.DataFrame, y_train:pd.DataFrame) -> SVC:
    # Initialize the model
    clf = SVC(random_state=3, probability=True)

    logging.info("Training the model")
    # Train the model
    clf.fit(X_train, y_train)

    save_model(dir=settings.MODEL.SVM, model=clf)

    return clf