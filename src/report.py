"""
Отчётность: графики (Matplotlib/Seaborn), Excel, PDF, отправка по email.
"""

import logging
import os
import smtplib
from email.message import EmailMessage
import matplotlib
matplotlib.use("Agg")  # работа без GUI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)

def plot_distributions(df: pd.DataFrame, columns: list = None, out_path: str = "reports/distributions.jpeg") -> str:
    """
    Построение гистограмм распределения числовых признаков.
    Функция возвращает путь к сохранённому файлу.
    """
    columns = columns or df.select_dtypes(include="number").columns.tolist()
    n = len(columns)
    if n == 0:
        raise ValueError("Нет числовых колонок для визуализации")
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    axes = [axes] if n == 1 else list(axes)
    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Распределение: {col}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("График распределений сохранён: %s", out_path)
    return out_path

def save_excel_report(stats: pd.DataFrame, out_path: str = "reports/report.xlsx",
                      sheet_name: str = "Статистика") -> str:
    """
    Вывод статистики и данных в Excel-файл через openpyxl.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        stats.to_excel(writer, sheet_name=sheet_name)
    logger.info("Excel-отчёт сохранён: %s", out_path)
    return out_path

def save_pdf_report(stats: pd.DataFrame, out_path: str = "reports/report.pdf",
                    title: str = "Отчёт по анализу данных") -> str:
    """
    Формирование PDF-отчёта с таблицей статистик через ReportLab.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph)
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError as e:
        raise ImportError("Установите reportlab: pip install reportlab") from e
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    data = [list(stats.columns)] + [[f"{v:.4f}" if isinstance(v, float) else str(v)
                                     for v in row] for row in stats.itertuples(index=True, name=None)]
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D32F2F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(table)
    doc.build(story)
    logger.info("PDF-отчёт сохранён: %s", out_path)
    return out_path

def send_email(smtp_host: str, port: int, user: str, password: str,
               to: str, subject: str, body: str,
               attachments: list = None, use_tls: bool = True) -> None:
    """
    Отправка отчёта по email через SMTP (с вложениями).
    """
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    for path in attachments or []:
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application",
                               subtype="octet-stream",
                               filename=os.path.basename(path))
    with smtplib.SMTP(smtp_host, port) as server:
        if use_tls:
            server.starttls()
        server.login(user, password)
        server.send_message(msg)
    logger.info("Отчёт отправлен на %s", to)
