from .const import TARGET, TEST_SIZE, RANDOM_STATE

import pandas
import numpy as np

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


def preprocessing(df) -> pandas.DataFrame :
    """
    
    """

    df_clean = clean_dataframe(df)

    df_y = df_clean[TARGET]
    df_X = df_clean.drop([TARGET], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=TEST_SIZE, stratify=df_y, random_state=RANDOM_STATE)


    # separate all data type :
    X_test_binary = X_test[:, [1]]
    X_train_binary = X_train[:, [1]]
    X_test_categ = X_test[:, [ 7, 8]]
    X_train_categ = X_train[:, [ 7, 8]]
    X_test_num = X_test[:, [2, 3, 4, 6]]
    X_train_num = X_train[:, [2, 3, 4, 6]]

    label = LabelBinarizer()
    X_train_binary = label.fit_transform(X_train_binary)
    X_test_binary = label.transform(X_test_binary)

    one_hot = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
    )
    X_train_categ = one_hot.fit_transform(X_train_categ)
    X_test_categ = one_hot.transform(X_test_categ)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    X_train = np.concatenate(
        [X_train_binary, X_train_categ, X_train_num],
        axis=1
    )
    X_test = np.concatenate(
        [X_test_binary, X_test_categ, X_test_num],
        axis=1
    )

    return X_train, X_test, y_train, y_test