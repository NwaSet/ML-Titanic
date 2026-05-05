from .const import TARGET

import pandas

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler


def clean_dataframe(df) -> pandas.DataFrame :
    """
    clean all data in the dataframe and return a new clean dataframe
    """

    df_clean = df.dropna(subset=["embarked"])

    df_clean["fare"] = df_clean["fare"].fillna(df_clean["fare"].mean())

    df_clean["age"] = df_clean["age"].fillna(
        df_clean.groupby(["pclass", "sex"])["age"].transform("mean")
    )

    df_clean["deck"] = df_clean["cabin"].str[0]
    df_clean = df_clean.drop(columns=["cabin"])
    df_clean = df_clean[df_clean['deck'] != 'T']

    df_clean = df_clean[df_clean["fare"] <= 400]

    return df_clean