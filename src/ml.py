"""
Построение и оценка базовых ML-моделей: регрессия и классификация.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

def train_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                     random_state: int = 42, model=None) -> dict:
    """
    Обучаение регрессионной модели,в качестве результата выполнение -
    метрики качества: RMSE, MAE, R².
    """
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=random_state)
    model = model or LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }
    logger.info("Регрессия: RMSE=%.4f, MAE=%.4f, R2=%.4f",metrics["RMSE"], metrics["MAE"], metrics["R2"])
    return {"model": model, "metrics": metrics, "y_test": y_test, "prediction": pred}

def train_classifier(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                     random_state: int = 42, model=None, average: str = "binary") -> dict:
    """Обучение классификатора, результат выполнения -
    метрики качества: accuracy, precision, recall, F1, ROC-AUC (для бинарной задачи).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)
    model = model or LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, pred, average=average, zero_division=0)),
        "F1": float(f1_score(y_test, pred, average=average, zero_division=0)),
    }
    if y.nunique() == 2 and hasattr(model, "predict_proba"):
        metrics["ROC_AUC"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    logger.info("Классификация: %s", {k: round(v, 4) for k, v in metrics.items()})
    return {"model": model, "metrics": metrics,"y_test": y_test, "prediction": pred}
