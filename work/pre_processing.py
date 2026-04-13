from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Titanic Dataset.csv"
TARGET_COLUMN = "survived"
TEST_SIZE = 0.2
RANDOM_STATE = 2


NUMERIC_FEATURES = [
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
    "ticket_group_size",
]

CATEGORICAL_FEATURES = [
    "pclass",
    "sex",
    "embarked",
    "deck",
    "title",
    "ticket_prefix",
    "is_alone",
]

TITLE_REPLACEMENTS = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def list_to_string(liste):
    """
    Convert a list to a comma-separated string.
    """
    if not isinstance(liste, list):
        raise TypeError("Le parametre doit etre une liste.")

    return ",".join(str(element) for element in liste)


def load_titanic_data(data_path=DATA_PATH):
    """
    Load the Titanic dataset.
    """
    return pd.read_csv(data_path)


def extract_title(name):
    """
    Extract a passenger title from the name column.
    """
    if pd.isna(name):
        return "Unknown"

    parts = str(name).split(",")
    if len(parts) < 2:
        return "Unknown"

    title_part = parts[1].split(".")[0].strip()
    title = TITLE_REPLACEMENTS.get(title_part, title_part)

    if title in COMMON_TITLES:
        return title

    return "Rare"


def extract_ticket_prefix(ticket):
    """
    Extract the textual prefix of a ticket.
    """
    if pd.isna(ticket):
        return "NO_PREFIX"

    cleaned_ticket = str(ticket).upper().replace(".", "").replace("/", "")
    parts = cleaned_ticket.split()
    prefix_parts = [part for part in parts if not part.isdigit()]

    if not prefix_parts:
        return "NO_PREFIX"

    return prefix_parts[0]


def engineer_titanic_features(df):
    """
    Clean raw Titanic data and create model-ready features.
    """
    df = df.copy()

    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])

    grouped_age = df.groupby(["pclass", "sex"])["age"].transform("median")
    df["age"] = df["age"].fillna(grouped_age)
    df["age"] = df["age"].fillna(df["age"].median())

    df["deck"] = df["cabin"].str.extract(r"([A-Za-z])", expand=False).fillna("Unknown")
    df["deck"] = df["deck"].replace({"T": "RareDeck"})

    df["title"] = df["name"].apply(extract_title)
    df["ticket_prefix"] = df["ticket"].apply(extract_ticket_prefix)
    df["ticket_group_size"] = df.groupby("ticket")["ticket"].transform("count")
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).map({True: "yes", False: "no"})

    df["pclass"] = df["pclass"].astype(str)

    return df.drop(columns=["name", "ticket", "cabin"])


def build_preprocessor():
    """
    Build the sklearn preprocessing pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def prepare_titanic_data(test_size=TEST_SIZE, random_state=RANDOM_STATE, data_path=DATA_PATH):
    """
    Load, clean, split and preprocess the Titanic dataset.
    """
    df_titanic = load_titanic_data(data_path)

    y_titanic = df_titanic[TARGET_COLUMN].astype(int)
    X_titanic = engineer_titanic_features(df_titanic.drop(columns=[TARGET_COLUMN]))

    X_train, X_test, y_train, y_test = train_test_split(
        X_titanic,
        y_titanic,
        test_size=test_size,
        stratify=y_titanic,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return {
        "df_titanic": df_titanic,
        "X_titanic": X_titanic,
        "y_titanic": y_titanic.to_numpy(),
        "column_names_X": X_titanic.columns.to_list(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "X_train_preprocessed": X_train_preprocessed,
        "X_test_preprocessed": X_test_preprocessed,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    data = prepare_titanic_data()

    print("Dataset brut :", data["df_titanic"].shape)
    print("Dataset apres feature engineering :", data["X_titanic"].shape)
    print("Train shape :", data["X_train"].shape)
    print("Test shape :", data["X_test"].shape)
    print("Train preprocessed shape :", data["X_train_preprocessed"].shape)
    print("Test preprocessed shape :", data["X_test_preprocessed"].shape)
