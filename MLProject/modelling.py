import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_recall_curve, auc, log_loss
)
import matplotlib.pyplot as plt
import os
import joblib
import argparse

# =========================
# ARGUMENT PARSER
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--data_path", type=str, default="fatalities_preprocessed")
args = parser.parse_args()

DATA_PATH = args.data_path
MODE = args.mode

# =========================
# SETUP OUTPUT
# =========================
os.makedirs("outputs", exist_ok=True)

# =========================
# LOAD DATA
# =========================
def load_data(path):
    X_train = pd.read_csv(f"{path}/X_train.csv").astype(float)
    X_test = pd.read_csv(f"{path}/X_test.csv").astype(float)
    y_train = pd.read_csv(f"{path}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{path}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data(DATA_PATH)
print("✅ Data loaded")

# =========================
# BUILD MODEL (GRID SEARCH)
# =========================
def build_model():
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    return grid

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba)
    }

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    metrics["pr_auc"] = auc(recall_curve, precision_curve)

    return metrics, y_pred, y_proba, precision_curve, recall_curve

# =========================
# ARTIFACT LOGGER
# =========================
def log_artifacts(y_test, y_pred, y_proba, precision_curve, recall_curve):

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("outputs/confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("outputs/roc_curve.png")
    plt.close()
    mlflow.log_artifact("outputs/roc_curve.png")

    # PR Curve
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("outputs/pr_curve.png")
    plt.close()
    mlflow.log_artifact("outputs/pr_curve.png")

    # Classification Report
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose()
    report_df.to_csv("outputs/classification_report.csv")
    mlflow.log_artifact("outputs/classification_report.csv")

# =========================
# TRAIN MODE
# =========================
if MODE == "train":

    with mlflow.start_run():  # 🔥 WAJIB

        grid = build_model()
        grid.fit(X_train, y_train)

        model = grid.best_estimator_

        metrics, y_pred, y_proba, precision_curve, recall_curve = evaluate_model(
            model, X_test, y_test
        )

        mlflow.log_params(grid.best_params_)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        log_artifacts(y_test, y_pred, y_proba, precision_curve, recall_curve)

        # Feature importance
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        feat_df.to_csv("outputs/feature_importance.csv", index=False)
        mlflow.log_artifact("outputs/feature_importance.csv")

        # 🔥 INI YANG PENTING
        mlflow.sklearn.log_model(
            model,
            name="model",
            pip_requirements=[
                "mlflow",
                "scikit-learn",
                "pandas",
                "numpy",
                "fastapi",
                "starlette<0.40.0"
            ]
        )

        print("\n=== TRAINING COMPLETE ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

# =========================
# EVALUATE MODE
# =========================
elif MODE == "evaluate":

    model = joblib.load("outputs/model.pkl")

    metrics, _, _, _, _ = evaluate_model(model, X_test, y_test)

    print("\n=== EVALUATION ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")