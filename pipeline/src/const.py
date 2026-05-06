DATA_PATH = "data/Titanic Dataset.csv"
MODEL_PATH = "pipeline/outputs/best_model.joblib"
OUTPUTS_PATH = "pipeline/outputs"
TRAINED_MODELS_PATH = "pipeline/outputs/models"

TARGET = "survived"

RANDOM_STATE = 42
TEST_SIZE = 0.2

SELECTED_COLUMNS = [
    "sex_female",
    "sex_male",
    "age",
    "fare",
    "embarked_C", "embarked_Q", "embarked_S",
    "deck_A", "deck_B", "deck_C", "deck_D",
    "deck_E", "deck_F", "deck_G", "deck_nan"
]
