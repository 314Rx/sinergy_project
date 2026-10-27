import argparse
import logging
import sys
import pandas as pd
import yaml
from src import analyze, clean, load, ml, report, validate


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("run.log", encoding="utf-8")])


def run_script(config: dict) -> dict:
    """
    Запуск: загрузка -> валидация -> очистка -> анализ -> ML -> отчёты.
    """
    setup_logging()

    """
    1. Загрузка данных
    """
    source = config["source"]
    if source["type"] == "csv":
        df = load.load_csv(source["path"], **source.get("kwargs", {}))
    elif source["type"] == "excel":
        df = load.load_excel(source["path"], **source.get("kwargs", {}))
    elif source["type"] == "sql":
        df = load.load_sql(source["query"], source["connection_string"])
    elif source["type"] == "api":
        df = load.load_api(source["url"], params=source.get("params"), data_key=source.get("data_key"))
    else:
        raise ValueError(f"Неизвестный тип источника: {source['type']}")

    """
    2. Валидация
    """
    validation_report = validate.validate(df, expected_dtypes=config.get("expected_dtypes"))
    pd.Series({k: v["count"] for k, v in validation_report["missing"].items()}).to_csv(
        "reports/missing_report.csv")

    """
    3. Очистка
    """
    df = clean.drop_duplicates(df)
    df = clean.handle_missing(df, strategy=config.get("missing_strategy", "median"))
    if config.get("date_columns"):
        df = clean.parse_dates(df, config["date_columns"], dayfirst=config.get("dayfirst", False))

    """
    4. Анализ
    """
    stats = analyze.basic_stats(df)
    print("\nБазовые статистики")
    print(stats)

    """
    5. ML
    """
    ml_results = None
    target = config.get("target")
    task = config.get("task")  # regression | classification | None
    if target and task:
        df_ml = clean.encode_categorical(df)
        X = df_ml.drop(columns=[target])
        X = X.select_dtypes(include="number")
        y = df_ml[target]
        if task == "regression":
            ml_results = ml.train_regression(X, y)
        elif task == "classification":
            ml_results = ml.train_classifier(X, y)
        print("\nМетрики модели")
        print(ml_results["metrics"])

    """
    6. Отчётность
    """
    plot_path = report.plot_distributions(df)
    excel_path = report.save_excel_report(stats)
    pdf_path = report.save_pdf_report(stats)

    """
    7. Отправка по email (опционально)
    """
    email = config.get("email")
    if email:
        report.send_email(attachments=[plot_path, excel_path, pdf_path], **email)
    return {"stats": stats, "ml": ml_results, "reports": {"plot": plot_path, "excel": excel_path, "pdf": pdf_path}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматизация обработки данных")
    parser.add_argument("--config", default="config/config.yaml", help="Путь к YAML-конфигу")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    result = run_script(config)
    logging.getLogger(__name__).info("Скрипт выполнен успешно!")


if __name__ == "__main__":
    main()
