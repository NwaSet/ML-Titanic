DATA_PATH = "data/Titanic Dataset.csv"
MODEL_PATH = "pipelin/outputs/best_model.joblib"

TARGET = "survived"

RANDOM_STATE = 42
TEST_SIZE = 0.2

SELECTED_COLUMNS = [
    "Sex",
    "age",
    "fare",
    "embarked_C", "embarked_Q", "embarked_S",
    "deck_A", "deck_B", "deck_C", "deck_D",
    "deck_E", "deck_F", "deck_G", "deck_nan"
]
