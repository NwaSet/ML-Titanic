import pandas

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from const import RANDOM_STATE
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


def train_models(selected_datasets, y_train, y_test):
    results = []

    models = get_models()

    for selection_name, data in selected_datasets.items():
        X_train = data["X_train"]
        X_test = data["X_test"]

        for model_name, model_config in models.items():
            grid = GridSearchCV(
                estimator=model_config["model"],
                param_grid=model_config["params"],
                cv=5,
                scoring="f1",
                n_jobs=1
            )

            grid.fit(X_train, y_train.values.ravel())

            best_model = grid.best_estimator_

            result = evaluate_model(
                best_model,
                X_train,
                X_test,
                y_train,
                y_test,
                selection_name,
                model_name
            )

            result["Best Parameters"] = str(grid.best_params_)

            results.append(result)

    return pandas.DataFrame(results)
