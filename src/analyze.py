"""
Анализ данных: статистики, временные ряды, аномалии.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица базовых статистик: mean, median, mode, std, min, max.
    """
    numeric = df.select_dtypes(include=np.number)
    stats = pd.DataFrame({
        "mean": numeric.mean(),
        "median": numeric.median(),
        "mode": numeric.mode().iloc[0] if not numeric.mode().empty else np.nan,
        "std": numeric.std(),
        "min": numeric.min(),
        "max": numeric.max(),
    }).round(4)
    logger.info("Рассчитаны базовые статистики для %d числовых колонок", len(stats))
    return stats


def time_series_decomposition(series: pd.Series, period: int = 12) -> pd.DataFrame:
    """
    Классическая декомпозиция временного ряда без внешних зависимостей.
    Тренд — скользящая средняя, сезонность — средние отклонения по фазам периода,
    остаток = исходный ряд - тренд - сезонность.
    """
    series = series.asfreq(series.index.to_period().freq or "D").interpolate()
    trend = series.rolling(window=period, center=True).mean()
    detrended = series - trend
    seasonal = detrended.groupby(detrended.index.to_period().to_timestamp().month
                                 if period == 12 else detrended.index % period).transform("mean")
    if period == 12:
        seasonal = detrended.groupby(detrended.index.month).transform("mean")
    else:
        seasonal = detrended.groupby(np.arange(len(detrended)) % period).transform("mean")
    residual = series - trend - seasonal
    result = pd.DataFrame({"original": series, "trend": trend,
                           "seasonal": seasonal, "residual": residual})
    logger.info("Декомпозиция ряда выполнена (period=%d)", period)
    return result


def detect_anomalies(series: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
    """
    Результат выполнения - Булева маска аномалий в временном ряде.
    """
    if method == "iqr":  # Метод Тьюки (IQR)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    elif method == "zscore":  # Метод Z-оценки
        std = series.std()
        mask = ((series - series.mean()) / std).abs() > threshold if std else pd.Series(False, index=series.index)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    logger.info("Аномалий найдено: %d из %d точек", int(mask.sum()), len(series))
    return mask
