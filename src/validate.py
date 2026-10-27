"""
Валидация данных при загрузке: дубликаты, пропуски, типы, выбросы.
Все результаты логируются, итог возвращается в виде словаря-отчёта.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_duplicates(df: pd.DataFrame, subset: list = None) -> dict:
    """
    Проверка наличия полных дубликатов строк.
    """
    n_dup = int(df.duplicated(subset=subset).sum())
    report = {"duplicates": n_dup, "total_rows": len(df)}
    logger.info("Проверка дубликатов: найдено %d из %d строк", n_dup, len(df))
    return report


def check_missing(df: pd.DataFrame) -> dict:
    """
    Подсчет пропусков по каждой колонке. Возвращает {колонка: {count, pct}}.
    """
    report = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        report[col] = {"count": n_missing, "pct": round(n_missing / len(df) * 100, 2) if len(df) else 0.0, }
    total = int(df.isna().sum().sum())
    logger.info("Проверка пропусков: всего %d пропущенных значений", total)
    return report


def check_dtypes(df: pd.DataFrame, expected: dict = None) -> dict:
    """
    Сравнение фактических типов с ожидаемыми.
    """
    report = {"actual": {c: str(t) for c, t in df.dtypes.items()}, "mismatches": []}
    if expected:
        for col, want in expected.items():
            if col not in df.columns:
                report["mismatches"].append({"column": col, "issue": "missing_column"})
            elif not np.issubdtype(df[col].dtype, np.dtype(want).type):
                report["mismatches"].append({"column": col, "expected": want, "actual": str(df[col].dtype)})
    if report["mismatches"]:
        logger.warning("Несоответствие типов: %s", report["mismatches"])
    else:
        logger.info("Типы данных корректны")
    return report


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None, k: float = 1.5) -> dict:
    """
    Выявление выбросов методом межквартильного размаха (IQR).
    Границы: [Q1 - k*IQR, Q3 + k*IQR]. Результат выполнения: маска и статистика по колонкам.
    """
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    report = {}
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
        report[col] = {
            "count": int(mask.sum()),
            "lower_bound": float(q1 - k * iqr),
            "upper_bound": float(q3 + k * iqr),
        }
        logger.info("IQR-выбросы в '%s': %d", col, report[col]["count"])
    return report


def detect_outliers_zscore(df: pd.DataFrame, columns: list = None, threshold: float = 3.0) -> dict:
    """
    Выявление выбросов методом Z-score (|z| > threshold).
    """
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    report = {}
    for col in columns:
        std = df[col].std()
        if std == 0 or pd.isna(std):
            report[col] = {"count": 0}
            continue
        z = (df[col] - df[col].mean()) / std
        report[col] = {"count": int((z.abs() > threshold).sum()), "threshold": threshold}
        logger.info("Z-score выбросы в '%s': %d", col, report[col]["count"])
    return report


def validate(df: pd.DataFrame, expected_dtypes: dict = None, subset_duplicates: list = None) -> dict:
    """
    Формирование единого отчёта, с логированием каждого этапа.
    """
    logger.info("Валидация данных (%d строк)", len(df))
    return {
        "duplicates": check_duplicates(df, subset_duplicates),
        "missing": check_missing(df),
        "dtypes": check_dtypes(df, expected_dtypes),
        "outliers_iqr": detect_outliers_iqr(df),
    }
