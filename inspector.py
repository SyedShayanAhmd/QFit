import pandas as pd
import io

def get_basic_stats(df):
    total_rows = len(df)
    total_cols = len(df.columns)
    return f"Rows: {total_rows}, Columns: {total_cols}, Columns names: {list(df.columns)}"

def get_inspection_html(df):
    buf = io.StringIO()
    buf.write("Dtypes:\n")
    buf.write(df.dtypes.to_string())
    buf.write("\n\nMissing values per column:\n")
    buf.write(df.isnull().sum().to_string())
    buf.write("\n\nDescribe (numeric):\n")
    buf.write(df.describe().to_string())
    return buf.getvalue()
