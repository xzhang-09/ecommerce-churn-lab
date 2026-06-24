import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV or Excel data into a pandas DataFrame.

    For .xlsx files with multiple sheets (e.g. the raw E-Commerce dataset,
    which has a "Data Dict" sheet alongside the actual data), the "E Comm"
    sheet is used if present; otherwise the first sheet is used.

    Args:
        file_path (str): Path to the CSV or Excel file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(file_path, sheet_name=None)
        return sheets.get("E Comm", next(iter(sheets.values())))

    return pd.read_csv(file_path)