# Автоматизация обработки данных

Python-скрипт для бизнес-аналитики: загрузка данных из разных источников,
валидация, очистка, анализ, ML-модели, автоматические отчёты (PDF/Excel/email).

## Быстрый старт

```bash
pip3 install -r requirements.txt
python main.py --config config.yaml
```

## Структура проекта

```
project/
├── main.py              # точка входа: скрипт для выполнения задания
├── config 
    ├── config.yaml      # конфигурация источника, ML, email
├── requirements.txt
├── src/
│   ├── load.py          # CSV, Excel, PostgreSQL (SQL с JOIN), REST API
│   ├── validate.py      # дубликаты, пропуски, типы, выбросы (IQR, Z-score)
│   ├── clean.py         # пропуски, дубли, One-Hot/Label, масштабирование, даты
│   ├── analyze.py       # статистики, декомпозиция временных рядов, аномалии
│   ├── ml.py            # регрессия и классификация + метрики
│   └── report.py        # графики, Excel, PDF, отправка по SMTP
├── tests/               # юнит-тесты
├── data/                # данные
└── reports/             # готовые отчёты

```

## Источники данных (`source.type`)

- `csv` — файлы CSV: `path`, `kwargs` (sep, encoding, parse_dates)
- `excel` — файлы .xlsx: `path`, `sheet_name`
- `sql` — PostgreSQL через SQLAlchemy: `connection_string`, `query` (поддержка JOIN, агрегатов)
- `api` — REST API: `url`, `params`, `data_key`

## ML

Задаётся в config.yaml:

```yaml
target: price        # целевая колонка
task: regression     # regression | classification
```

Регрессия → RMSE, MAE, R². Классификация → accuracy, precision, recall, F1, ROC-AUC.

## Планировщик (cron / Task Scheduler)

Ежедневный запуск в 07:30:

```cron
30 7 * * * cd /path/to/project && /usr/bin/python3 main.py --config config.yaml >> cron.log 2>&1
```

Windows (Task Scheduler): действие — запуск `python main.py`, триггер — ежедневно.

## Email-рассылка

Раскомментируйте блок `email` в config.yaml и укажите SMTP-настройки.

## Тестирование

```bash
python -m pytest tests/test.py -v
```

## Логирование

Все этапы пайплайна пишутся в консоль и файл `run.log`.
