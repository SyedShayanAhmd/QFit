"""
previewer.py

Provides DataPreviewFrame — a reusable frame that shows a DataFrame (treeview or text),
and simple operations such as show head, show dtypes.
"""
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext 
import pandas as pd
# IN YOUR previewer.py, UPDATE THE DataPreviewFrame CLASS

class DataPreviewFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Show head(50)", command=self._show_head).pack(side="left", padx=4, pady=4)
        ttk.Button(toolbar, text="Show dtypes", command=self._show_dtypes).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Show full table (text)", command=self._show_full_text).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Show info", command=self._show_info).pack(side="left", padx=4)
        
        # display area
        self.text = scrolledtext.ScrolledText(self, wrap="none", width=100, height=20)
        self.text.pack(fill="both", expand=True)
        self._df = None

    def set_dataframe(self, df: pd.DataFrame):
        self._df = df.copy()
        self._show_head()

    def set_text(self, txt: str):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", txt)

    def _show_head(self):
        if self._df is None:
            self.set_text("No data loaded")
            return
        try:
            # Show more rows and ensure proper formatting
            preview = self._df.head(50)
            txt = f"Shape: {self._df.shape}\nColumns: {list(self._df.columns)}\n\n"
            txt += preview.to_string(max_rows=50, max_cols=20)
            self.set_text(txt)
        except Exception as e:
            self.set_text(f"Error displaying data: {str(e)}")

    def _show_dtypes(self):
        if self._df is None:
            self.set_text("No data loaded")
            return
        txt = "Data Types:\n" + "="*50 + "\n"
        for col in self._df.columns:
            dtype = str(self._df[col].dtype)
            non_null = self._df[col].notna().sum()
            total = len(self._df[col])
            txt += f"{col:<20} {dtype:<15} {non_null}/{total} non-null\n"
        self.set_text(txt)

    def _show_full_text(self):
        if self._df is None:
            self.set_text("No data loaded")
            return
        try:
            # For large dataframes, show sample
            if len(self._df) > 100:
                txt = f"Data too large ({len(self._df)} rows). Showing first 100 rows:\n\n"
                txt += self._df.head(100).to_string()
            else:
                txt = self._df.to_string()
            self.set_text(txt)
        except Exception as e:
            self.set_text(f"Error displaying full data: {str(e)}")
            
    def _show_info(self):
        if self._df is None:
            self.set_text("No data loaded")
            return
        try:
            buffer = []
            buffer.append("DATA INFO SUMMARY")
            buffer.append("="*50)
            buffer.append(f"Shape: {self._df.shape}")
            buffer.append(f"Memory usage: {self._df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            buffer.append("\nCOLUMN INFO:")
            buffer.append("-"*30)
            
            for col in self._df.columns:
                dtype = self._df[col].dtype
                non_null = self._df[col].notna().sum()
                null_count = self._df[col].isna().sum()
                unique_count = self._df[col].nunique()
                
                buffer.append(f"{col}:")
                buffer.append(f"  Type: {dtype}")
                buffer.append(f"  Non-null: {non_null}/{len(self._df)} ({non_null/len(self._df)*100:.1f}%)")
                buffer.append(f"  Null: {null_count} ({null_count/len(self._df)*100:.1f}%)")
                buffer.append(f"  Unique: {unique_count}")
                
                if pd.api.types.is_numeric_dtype(self._df[col]):
                    buffer.append(f"  Min: {self._df[col].min()}")
                    buffer.append(f"  Max: {self._df[col].max()}")
                    buffer.append(f"  Mean: {self._df[col].mean():.2f}")
                    
            self.set_text("\n".join(buffer))
        except Exception as e:
            self.set_text(f"Error generating info: {str(e)}")