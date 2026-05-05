from const import TARGET, TEST_SIZE, RANDOM_STATE

import pandas
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler


def list_to_string(liste):
    """
    Convertit une liste en chaîne de caractères séparée par des virgules.
    
    param liste: list - La liste à convertir
    return: str - La chaîne résultante
    """
    if not isinstance(liste, list):
        raise TypeError("Le paramètre doit être une liste.")
    
    # Conversion de chaque élément en chaîne pour éviter les erreurs
    return ",".join(str(element) for element in liste)


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


def preprocessing(df) -> tuple[pandas.DataFrame,pandas.DataFrame,pandas.DataFrame,pandas.DataFrame]:
    """
    will call clean_data and then preprocesse all data so they are usable for the models.
    """

    df_clean = clean_dataframe(df)

    df_y = df_clean[TARGET].to_numpy()
    df_clean = df_clean.drop([TARGET], axis=1)
    column_names_X = df_clean.keys().to_list()
    df_X = df_clean.to_numpy()


    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=TEST_SIZE, stratify=df_y, random_state=RANDOM_STATE)


    # separate all data type :

    # categorical
    X_train_categ = X_train[:, [2, 8, 9]]
    X_test_categ = X_test[:, [2, 8, 9]]

    # numeric
    X_train_num = X_train[:, [0, 3, 4, 5, 7]]
    X_test_num = X_test[:, [0, 3, 4, 5, 7]]

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
        [X_train_categ, X_train_num],
        axis=1
    )
    X_test = np.concatenate(
        [X_test_categ, X_test_num],
        axis=1
    )

    categ_original_cols = [
        column_names_X[2],
        column_names_X[8],
        column_names_X[9]
    ]

    categ_cols = one_hot.get_feature_names_out(categ_original_cols)

    num_cols = [
        column_names_X[0],
        column_names_X[3],
        column_names_X[4],
        column_names_X[5],
        column_names_X[7]
    ]

    all_columns = list(categ_cols) + list(num_cols)

    all_columns = list(categ_cols) + list(num_cols)

    df_X_train = pandas.DataFrame(X_train, columns=all_columns)
    df_X_test = pandas.DataFrame(X_test, columns=all_columns)

    df_y_train = pandas.DataFrame(y_train, columns=["survived"])
    df_y_test = pandas.DataFrame(y_test, columns=["survived"])

    return df_X_train, df_X_test, df_y_train, df_y_test