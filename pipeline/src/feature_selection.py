from const import SELECTED_COLUMNS

import pandas

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection(
        df_X_train : pandas.DataFrame,
        df_X_test : pandas.DataFrame,
        df_y_train : pandas.DataFrame
        ) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame] :
    """
    
    """

    # selection perso :
    df_X_train_P = df_X_train[SELECTED_COLUMNS]
    df_X_test_P = df_X_test[SELECTED_COLUMNS]


    selector = VarianceThreshold(threshold=0.05)

    select_X_train_VT = selector.fit_transform(df_X_train)
    train_selected_columns = df_X_train.columns[selector.get_support()]

    select_X_test_VT = selector.transform(df_X_test)
    test_selected_columns = train_selected_columns

    df_X_train_VT = pandas.DataFrame(select_X_train_VT, columns=train_selected_columns)
    df_X_test_VT = pandas.DataFrame(select_X_test_VT, columns=test_selected_columns)


    selector = SelectKBest(score_func=f_classif, k=7)

    select_X_train_KB = selector.fit_transform(df_X_train, df_y_train.values.ravel())
    train_selected_columns = df_X_train.columns[selector.get_support()]

    select_X_test_KB = selector.transform(df_X_test)
    test_selected_columns = train_selected_columns

    df_X_train_KB = pandas.DataFrame(select_X_train_KB, columns=train_selected_columns)
    df_X_test_KB = pandas.DataFrame(select_X_test_KB, columns=test_selected_columns)


    return df_X_train_P, df_X_test_P, df_X_train_VT, df_X_test_VT, df_X_train_KB, df_X_test_KB