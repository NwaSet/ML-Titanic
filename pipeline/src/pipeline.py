from load_data import load_data
from preprocessing import preprocessing

## 1. load data in a dataframe 

df_titanic = load_data()

## 2. preprocessing and cleaning data

df_X_train, df_X_test, df_y_train, df_y_test = preprocessing(df_titanic)