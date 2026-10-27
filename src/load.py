import pandas as pd
import requests
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Загрузка данных из csv-файла
    """
    df = pd.read_csv(path, **kwargs)
    logger.info("CSV '%s': загружено %d строк, %d колонок", path, *df.shape)
    return df


def load_excel(path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
    """
    Загрузка данных из Excel-файла
    """
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    logger.info("Excel '%s' (лист %s): загружено %d строк, %d колонок", path, sheet_name, *df.shape)
    return df


def load_sql(query: str, connection_string: str = None, engine=None, **kwargs) -> pd.DataFrame:
    """
    Выполнение SQL-запроса
    connection_string следующего вида: postgresql://user:pass@host:5432/db
    """
    if engine is None:
        if not connection_string:
            raise ValueError("Нужно передать либо connection_string, либо engine")
        engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, **kwargs)
    logger.info("SQL: запрос вернул %d строк, %d колонок", *df.shape)
    return df


def load_api(url: str, params: dict = None, timeout: int = 30, data_key: str = None, **kwargs) -> pd.DataFrame:
    """
    Загрузка данных из REST API и преобразование JSON в DataFrame.
    """
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    records = payload[data_key] if data_key else payload
    df = pd.json_normalize(records)
    logger.info("API '%s': получено %d записей", url, len(df))
    return df
