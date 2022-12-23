import pandas as pd
from conf.conf import logging

def get_data(link: str) -> pd.DataFrame:
    """
    This data gets data from postgres
    """
    logging.info("Extracting df")
    df = pd.read_csv(link)
    logging.info("DF is extracted")

    return df