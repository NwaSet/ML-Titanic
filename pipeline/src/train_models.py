from pathlib import Path

import joblib
import pandas

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from const import MODEL_PATH, RANDOM_STATE, TRAINED_MODELS_PATH
from evaluate import evaluate_model


def get_models():
    return {
        "DT": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "RF": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
    }


def get_model_file_path(selection_name, model_name):
    return Path(TRAINED_MODELS_PATH) / f"{model_name}_{selection_name}.joblib"


def train_or_load_model(model_config, X_train, y_train, selection_name, model_name):
    model_file_path = get_model_file_path(selection_name, model_name)
    model_file_path.parent.mkdir(parents=True, exist_ok=True)

    if model_file_path.exists():
        print(f"Load existing model: {model_file_path}")
        return joblib.load(model_file_path)

    print(f"Train model: {model_name} + {selection_name}")

    grid = GridSearchCV(
        estimator=model_config["model"],
        param_grid=model_config["params"],
        cv=5,
        scoring="f1",
        n_jobs=1
    )

    y_train_array = y_train.values.ravel()
    grid.fit(X_train, y_train_array)

    best_model = grid.best_estimator_

    cv_accuracy_scores = cross_val_score(
        best_model,
        X_train,
        y_train_array,
        cv=5,
        scoring="accuracy"
    )

    cv_f1_scores = cross_val_score(
        best_model,
        X_train,
        y_train_array,
        cv=5,
        scoring="f1"
    )

    model_data = {
        "model": best_model,
        "best_params": grid.best_params_,
        "cv_accuracy_scores": cv_accuracy_scores,
        "cv_f1_scores": cv_f1_scores,
    }

    joblib.dump(model_data, model_file_path)
    return model_data


def train_models(selected_datasets, y_train, y_test):
    results = []

    models = get_models()

    for selection_name, data in selected_datasets.items():
        X_train = data["X_train"]
        X_test = data["X_test"]

        for model_name, model_config in models.items():
            model_data = train_or_load_model(
                model_config,
                X_train,
                y_train,
                selection_name,
                model_name
            )

            result = evaluate_model(
                model_data["model"],
                X_train,
                X_test,
                y_train,
                y_test,
                selection_name,
                model_name,
                model_data["cv_accuracy_scores"],
                model_data["cv_f1_scores"]
            )

            result["Best Parameters"] = str(model_data["best_params"])

            results.append(result)

    df_results = pandas.DataFrame(results)

    best_result = df_results.sort_values("F1 Score", ascending=False).iloc[0]
    best_model_data = joblib.load(
        get_model_file_path(best_result["Selection"], best_result["Model"])
    )

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_data, MODEL_PATH)

    return df_results
