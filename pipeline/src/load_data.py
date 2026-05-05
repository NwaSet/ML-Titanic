from .const import *

import pandas

def load_data() -> pandas.DataFrame:
    """
    return a pandas dataframe
    """

    df = pandas.read_csv(DATA_PATH)
    return df