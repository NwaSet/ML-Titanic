from load_data import load_data
from preprocessing import preprocessing
from feature_selection import feature_selection
from train_models import train_models

## 1. load data in a dataframe 

df_titanic = load_data()

## 2. preprocessing and cleaning data

df_X_train, df_X_test, df_y_train, df_y_test = preprocessing(df_titanic)

## 3. feature selection

df_X_train_P, df_X_test_P, df_X_train_VT, df_X_test_VT, df_X_train_KB, df_X_test_KB = feature_selection(df_X_train, df_X_test, df_y_train)

selected_datasets = {
    "P": {
        "X_train": df_X_train_P,
        "X_test": df_X_test_P,
    },
    "VT": {
        "X_train": df_X_train_VT,
        "X_test": df_X_test_VT,
    },
    "KB": {
        "X_train": df_X_train_KB,
        "X_test": df_X_test_KB,
    },
}

results = train_models(selected_datasets, df_y_train, df_y_test)

results.to_csv("pipeline/outputs/model_accuracy.csv", index=False)

print(results.sort_values("F1 Score", ascending=False))