from sklearn.model_selection import train_test_split
from conf.conf import logging
import pandas as pd


def split(df: pd.DataFrame) -> pd.DataFrame:
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