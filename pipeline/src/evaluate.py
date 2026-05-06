import pandas

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def evaluate_model(model, X_train, X_test, y_train, y_test, selection_name, model_name):
    y_train_array = y_train.values.ravel()
    y_test_array = y_test.values.ravel()

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cv_accuracy_scores = cross_val_score(
        model,
        X_train,
        y_train_array,
        cv=5,
        scoring="accuracy"
    )

    cv_f1_scores = cross_val_score(
        model,
        X_train,
        y_train_array,
        cv=5,
        scoring="f1"
    )

    tn, fp, fn, tp = confusion_matrix(y_test_array, y_test_pred).ravel()

    custom_accuracy = (tn + tp + fn) / (tn + fp + fn + tp)

    return {
        "Selection": selection_name,
        "Model": model_name,
        "Train Accuracy": round(accuracy_score(y_train_array, y_train_pred), 4),
        "Test Accuracy": round(accuracy_score(y_test_array, y_test_pred), 4),
        "CV Accuracy Mean": round(cv_accuracy_scores.mean(), 4),
        "CV Accuracy Std": round(cv_accuracy_scores.std(), 4),
        "CV F1 Mean": round(cv_f1_scores.mean(), 4),
        "CV F1 Std": round(cv_f1_scores.std(), 4),
        "F1 Score": round(f1_score(y_test_array, y_test_pred), 4),
        "Precision": round(precision_score(y_test_array, y_test_pred), 4),
        "Recall": round(recall_score(y_test_array, y_test_pred), 4),
        "Custom Accuracy": round(custom_accuracy, 4),
    }
