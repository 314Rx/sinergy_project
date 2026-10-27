"""
Очистка данных: пропуски, дубликаты, кодирование, масштабирование, даты.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def handle_missing(df: pd.DataFrame, strategy: str = "median", columns: list = None) -> pd.DataFrame:
    """
    Обрабобтка пропусков.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    for col in columns:
        n = int(df[col].isna().sum())
        if n == 0:
            continue
        if strategy == "mean":
            fill = df[col].mean()
        elif strategy == "median":
            fill = df[col].median()
        elif strategy == "mode":
            fill = df[col].mode(dropna=True).iloc[0]
        elif strategy == "drop":
            df = df.dropna(subset=[col])
            logger.info("'%s': удалено %d строк с пропусками", col, n)
            continue
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")
        df[col] = df[col].fillna(fill)
        logger.info(
            "'%s': заполнено %d пропусков значением %s (%s)",
            col, n, fill, strategy)
    return df


def drop_duplicates(df: pd.DataFrame, subset: list = None, keep: str = "first") -> pd.DataFrame:
    """
    Удаление дубликатов строк.
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    logger.info("Удалено дубликатов: %d", before - len(df))
    return df


def encode_categorical(df: pd.DataFrame, columns: list = None,
                       method: str = "onehot", drop_first: bool = True) -> pd.DataFrame:
    """
    Кодировка категориальных признаков.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    if method == "onehot":
        encoder = OneHotEncoder(drop="first" if drop_first else None, sparse_output=False, handle_unknown="ignore")
        if columns:
            arr = encoder.fit_transform(df[columns])
            names = encoder.get_feature_names_out(columns)
            encoded = pd.DataFrame(arr, columns=names, index=df.index)
            df = pd.concat([df.drop(columns=columns), encoded], axis=1)
            logger.info("One-Hot Encoding применён к %s", columns)
    elif method == "label":
        for col in columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            logger.info("Label Encoding применён к '%s'", col)
    else:
        raise ValueError(f"Неизвестный метод кодирования: {method}")
    return df


def scale_numeric(df: pd.DataFrame, columns: list = None,
                  method: str = "standard") -> pd.DataFrame:
    """
    Масштабирование числовых признаков.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.info("Масштабирование (%s) применено к %d колонкам", method, len(columns))
    return df


def parse_dates(df: pd.DataFrame, columns: list, dayfirst: bool = False, errors: str = "coerce") -> pd.DataFrame:
    """
    Преобразование строковых дат в формат datetime.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, errors=errors)
            logger.info("'%s' преобразована в datetime", col)
    return df
