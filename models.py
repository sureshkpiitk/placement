from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

from xgboost import XGBClassifier


def fit_and_evaluate(pipeline: Pipeline, X_train, X_test, y_train, y_test) -> Tuple[Dict[str, Any], Pipeline]:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    labels = np.unique(y_test)
    avg = 'binary' if len(labels) == 2 else 'macro'
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, preds, average=avg, zero_division=0)),
        "f1": float(f1_score(y_test, preds, average=avg, zero_division=0)),
        "auc": float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])),
        "mcc": float(matthews_corrcoef(y_test, preds)),
    }
    return metrics, pipeline


def train_models(preprocessor, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    results = {}

    # Logistic Regression
    pipe_lr = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    metrics_lr, model_lr = fit_and_evaluate(pipe_lr, X_train, X_test, y_train, y_test)
    results['logistic_regression'] = {'metrics': metrics_lr, 'model': model_lr}
    save_model(model_lr, "models/logistic_regression.joblib")

    # Decision Tree
    pipe_dt = Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(random_state=42))])
    metrics_dt, model_dt = fit_and_evaluate(pipe_dt, X_train, X_test, y_train, y_test)
    results['decision_tree'] = {'metrics': metrics_dt, 'model': model_dt}
    save_model(model_dt, "models/decision_tree.joblib")

    # K-Nearest Neighbors
    pipe_knn = Pipeline([("pre", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=5))])
    metrics_knn, model_knn = fit_and_evaluate(pipe_knn, X_train, X_test, y_train, y_test)
    results['knn'] = {'metrics': metrics_knn, 'model': model_knn}
    save_model(model_knn, "models/knn.joblib")

    # Naive Bayes (Gaussian)
    pipe_nb = Pipeline([("pre", preprocessor), ("clf", GaussianNB())])
    metrics_nb, model_nb = fit_and_evaluate(pipe_nb, X_train, X_test, y_train, y_test)
    results['naive_bayes'] = {'metrics': metrics_nb, 'model': model_nb}
    save_model(model_nb, "models/naive_bayes.joblib")

    # Random Forest
    pipe_rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    metrics_rf, model_rf = fit_and_evaluate(pipe_rf, X_train, X_test, y_train, y_test)
    results['random_forest'] = {'metrics': metrics_rf, 'model': model_rf}
    save_model(model_rf, "models/random_forest.joblib")

    # XGBoost
    pipe_xgb = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    metrics_xgb, model_xgb = fit_and_evaluate(pipe_xgb, X_train, X_test, y_train, y_test)
    results['xgboost'] = {'metrics': metrics_xgb, 'model': model_xgb}
    save_model(model_xgb, "models/xgboost.joblib")

    return results


def pick_best(results: Dict[str, Any]) -> Tuple[str, Pipeline, Dict[str, Any]]:
    best_name = None
    best_score = -1.0
    for name, info in results.items():
        acc = info['metrics'].get('accuracy') or 0.0
        if acc > best_score:
            best_score = acc
            best_name = name
    return best_name, results[best_name]['model'], results[best_name]['metrics']


def save_model(model: Pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
