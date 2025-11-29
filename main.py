# main.py - Updated with working folder browser and directory viewer
"""
main.py - Updated with working folder browser and directory viewer
"""
import multiprocessing
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import threading
import matplotlib
matplotlib.use("Agg")
from utils import ensure_dir, safe_filename
from data_loader import load_file_or_url, browse_and_preview_folder
from inspector import get_basic_stats, get_inspection_html
from results_visualizer import AdvancedResultsVisualizer
from visualizer import Visualizer
from preprocess import Preprocessor
from previewer import DataPreviewFrame
from utils import ensure_dir, safe_filename
from advanced_editor import AdvancedEditor
from preprocess import AdvancedPreprocessor
import traceback
from interactive_plot_viewer import InteractivePlotViewer
from enhanced_preprocessor import EnhancedPreprocessor
# Add this with your other imports at the top of main.py
try:
    from enhanced_visualizer import EnhancedVisualizer
except ImportError as e:
    print(f"Warning: Could not import EnhancedVisualizer: {e}")
    EnhancedVisualizer = None
# add this near other trainer imports
try:
    from trainer_pycaret import PyCaretTrainer
except Exception:
    PyCaretTrainer = None

import sys

# Determine if running as a script or frozen exe
if getattr(sys, 'frozen', False):
    # If run as .exe, use the directory of the executable
    APP_DIR = os.path.dirname(sys.executable)
else:
    # If run as script, use the file location
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Define output directory relative to the executable location
OUTPUT_DIR = os.path.join(APP_DIR, "output")
ensure_dir(OUTPUT_DIR)

import threading
import queue

class Tooltip:
    """Simple tooltip for tkinter widgets. Usage: Tooltip(widget, "help text")"""
    def __init__(self, widget, text, delay=400):
        self.root = root
        self.thread_safe = ThreadSafeTkinter(root)
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self._tip = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")

    def _enter(self, event=None):
        self._schedule()

    def _leave(self, event=None):
        self._unschedule()
        self._hide_tip()

    def _schedule(self):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._show_tip)

    def _unschedule(self):
        if self._id:
            try:
                self.widget.after_cancel(self._id)
            except Exception:
                pass
            self._id = None

    def _show_tip(self):
        if self._tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self._tip, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4, ipady=2)

    def _hide_tip(self):
        try:
            if self._tip:
                self._tip.destroy()
        finally:
            self._tip = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("QFit — Ribbon UI - By Syed Shayan Ahmed")
        root.geometry("1100x700")
        root.minsize(900, 600)

        # Initialize thread-safe Tkinter FIRST

        
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Data containers
        self.df_original = None
        self.preview_df = None
        self.df_processed = None
        self.filepath = None

        # Preprocessor
        self.preprocessor = None

        # ADD THIS LINE: Make OUTPUT_DIR available as instance attribute
        self.OUTPUT_DIR = OUTPUT_DIR

        # Menu and Notebook
        self._build_menu()
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tabs
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_preproc = ttk.Frame(self.notebook)
        self.tab_ml = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_data, text="Data")
        self.notebook.add(self.tab_preproc, text="Preprocess")
        self.notebook.add(self.tab_ml, text="ML Application")
        self.notebook.add(self.tab_results, text="Results")

        # Build UIs
        self._build_data_tab()
        self._build_preproc_tab()
        self._build_ml_tab()
        self._build_results_tab()

        # Status bar
        self.status = ttk.Label(root, text="Ready", anchor="w")
        self.status.grid(row=1, column=0, sticky="ew")

        # Progress bar for downloadsF
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, sticky="ew")

    # ==================== SESSION MANAGEMENT METHODS ====================
    def _save_analysis_session(self):
            """Save complete analysis: Data, Model, Settings, and PLOTS + RULES in one folder."""
            # 1. Check if data exists
            if self.df_original is None:
                messagebox.showinfo("No Data", "Load data first.")
                return
                
            # 2. Ask user for location
            target_folder = filedialog.askdirectory(title="Select Folder to Save Session")
            if not target_folder:
                return
                
            import shutil
            import traceback
            from sklearn.pipeline import Pipeline
            from sklearn.tree import export_text
            
            try:
                self.status.config(text="Saving session...")
                
                # --- A. Save Data ---
                self.df_original.to_csv(os.path.join(target_folder, "original_data.csv"), index=False)
                if self.df_processed is not None:
                    self.df_processed.to_csv(os.path.join(target_folder, "processed_data.csv"), index=False)
                    
                # --- B. Save Settings ---
                self._save_preprocessing_settings(os.path.join(target_folder, "preprocessing_settings.txt"))
                
                # --- C. Save ML Results & Model ---
                if hasattr(self, 'last_pycaret_results'):
                    self._save_ml_results(os.path.join(target_folder, "ml_results.txt"))
                    
                    # Save .pkl
                    artifacts = self.last_pycaret_results.get("artifacts", {})
                    model_path_source = artifacts.get("model_path")
                    if model_path_source:
                        # PyCaret sometimes omits extension in path string
                        candidates = [model_path_source, model_path_source + ".pkl"]
                        for src in candidates:
                            if os.path.exists(src):
                                shutil.copy2(src, os.path.join(target_folder, "trained_model.pkl"))
                                break

                # --- D. Save Plots AND Rules (Together) ---
                plots_dir = os.path.join(target_folder, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                # 1. Save Images
                if hasattr(self, 'generated_plots') and self.generated_plots:
                    for plot_name, plot_path in self.generated_plots.items():
                        if os.path.exists(plot_path):
                            shutil.copy2(plot_path, os.path.join(plots_dir, os.path.basename(plot_path)))

                # 2. Generate and Save Rules Text (Same Folder)
                if hasattr(self, 'last_pycaret_results'):
                    model = self.last_pycaret_results.get('model')
                    features = self.last_pycaret_results.get('features', [])
                    
                    # Unwrap Pipeline
                    if isinstance(model, Pipeline):
                        if 'actual_estimator' in model.named_steps:
                            model = model.named_steps['actual_estimator']
                        else:
                            model = model.steps[-1][1]

                    content = f"Model Logic for: {type(model).__name__}\n"
                    content += "=" * 60 + "\n\n"
                    
                    try:
                        # Logic for Linear Models
                        if hasattr(model, 'coef_'):
                            content += "=== EQUATION ===\n"
                            intercept = getattr(model, 'intercept_', 0)
                            if isinstance(intercept, (list, np.ndarray)): intercept = intercept[0]
                            
                            eq_parts = [f"{intercept:.4f}"]
                            coefs = model.coef_
                            if coefs.ndim > 1: coefs = coefs[0]
                            
                            for i, c in enumerate(coefs):
                                if i < len(features):
                                    sign = "+" if c >= 0 else "-"
                                    eq_parts.append(f"\n   {sign} ({abs(c):.5f} * {features[i]})")
                            content += f"y = {''.join(eq_parts)}"

                        # Logic for Tree Models
                        elif hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                            content += "=== DECISION TREE STRUCTURE ===\n"
                            est = model
                            if hasattr(model, 'estimators_'): est = model.estimators_[0]
                            content += export_text(est, feature_names=features[:est.n_features_in_] if len(features) >= est.n_features_in_ else None)
                        
                        else:
                            content += "No text-based logic available for this algorithm.\nSee feature_importance.png for details."

                        # Write the file into the PLOTS folder
                        with open(os.path.join(plots_dir, "model_structure_and_rules.txt"), "w", encoding="utf-8") as f:
                            f.write(content)
                            
                    except Exception as e:
                        print(f"Could not save rules text: {e}")

                # --- E. Save Logs ---
                if hasattr(self, 'ml_logs_text'):
                    with open(os.path.join(target_folder, "training_logs.txt"), "w", encoding="utf-8") as f:
                        f.write(self.ml_logs_text.get("1.0", "end"))

                self.status.config(text=f"Session saved to: {target_folder}")
                messagebox.showinfo("Success", f"Saved to: {target_folder}\n\nPlots and Rules are located in the 'plots' subfolder.")
                self._open_folder(target_folder)
                
            except Exception as e:
                messagebox.showerror("Save Failed", f"Error: {str(e)}\n{traceback.format_exc()}")

    def _save_preprocessing_settings(self, filepath):
        """Save preprocessing settings to a text file"""
        with open(filepath, 'w') as f:
            f.write("PREPROCESSING SETTINGS\n")
            f.write("=" * 50 + "\n\n")
            
            # Global settings
            f.write("GLOBAL SETTINGS:\n")
            f.write(f"Target Column: {getattr(self, 'target_combo', 'Not set').get()}\n")
            f.write(f"Encode Method: {getattr(self, 'encode_method', 'Not set').get()}\n")
            f.write(f"Transpose Data: {getattr(self, 'transpose_var', 'Not set').get()}\n")
            f.write(f"Replace Negatives: {getattr(self, 'remove_neg_var', 'Not set').get()}\n\n")
            
            # Per-column settings
            f.write("PER-COLUMN SETTINGS:\n")
            f.write("-" * 30 + "\n")
            
            if hasattr(self, 'per_col_widgets'):
                for col, widgets in self.per_col_widgets.items():
                    f.write(f"\nColumn: {col}\n")
                    f.write(f"  Missing Strategy: {widgets['strategy'].get()}\n")
                    f.write(f"  Fill Value: {widgets['fill_value'].get()}\n")
                    f.write(f"  Min Clamp: {widgets['low'].get()}\n")
                    f.write(f"  Max Clamp: {widgets['high'].get()}\n")
                    f.write(f"  OneHot: {widgets['onehot'].get()}\n")
                    f.write(f"  Label Encode: {widgets['label'].get()}\n")
                    
                    if widgets['custom_mapping'].get():
                        f.write(f"  Custom Mapping: {widgets['custom_mapping'].get()}\n")
                    if widgets['bins'].get():
                        f.write(f"  Bins: {widgets['bins'].get()}\n")
                    if widgets['bin_labels'].get():
                        f.write(f"  Bin Labels: {widgets['bin_labels'].get()}\n")

    def _save_ml_results(self, filepath):
        """Save ML results to a text file"""
        with open(filepath, 'w') as f:
            f.write("MACHINE LEARNING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            if hasattr(self, 'last_pycaret_results'):
                results = self.last_pycaret_results
                f.write(f"Problem Type: {getattr(self, 'problem_type', 'Unknown').get()}\n")
                f.write(f"Experiment Type: {getattr(self, 'experiment_type', 'Unknown').get()}\n\n")
                
                f.write("METRICS:\n")
                for metric, value in results.get('metrics', {}).items():
                    f.write(f"  {metric}: {value}\n")
                
                if results.get('model'):
                    f.write(f"\nModel Type: {type(results['model']).__name__}\n")

    def _copy_output_files(self, session_folder):
        """Copy output files to session folder"""
        for file in os.listdir(self.OUTPUT_DIR):
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.csv', '.pkl')):
                src = os.path.join(self.OUTPUT_DIR, file)
                dst = os.path.join(session_folder, file)
                try:
                    import shutil
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"Could not copy {file}: {e}")

    def _open_folder(self, folder_path):
        """Open folder in file explorer"""
        try:
            if os.name == "nt":
                os.startfile(folder_path)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showinfo("Folder Location", f"Files saved to:\n{folder_path}")

    # ==================== MENU METHODS ====================
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open File or URL...", command=self._open_data_dialog)
        filemenu.add_command(label="Browse Folder...", command=self._browse_folder)
        filemenu.add_command(label="Save Analysis Session", command=self._save_analysis_session)
        filemenu.add_command(label="Show Output Folder", command=self._open_output_dir)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Quick Guide", command=self._show_help)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

    def _build_data_tab(self):
        # Main vertical paned window for the entire data tab
        main_paned = ttk.Panedwindow(self.tab_data, orient="vertical")
        main_paned.pack(fill="both", expand=True)

        # ==================== TOP SECTION: URL/Path Input ====================
        top_section = ttk.LabelFrame(main_paned, text="Data Source")
        top_section.pack(fill="x", padx=6, pady=6)
        
        # URL/Path input row
        input_frame = ttk.Frame(top_section)
        input_frame.pack(fill="x", padx=6, pady=6)
        
        ttk.Label(input_frame, text="Path or URL:").pack(side="left", padx=4, pady=4)
        self.path_var = tk.StringVar()
        entry = ttk.Entry(input_frame, textvariable=self.path_var, width=80)
        entry.pack(side="left", padx=4, pady=4, fill="x", expand=True)
        
        # Add tooltip with examples
        examples = (
            "Examples:\n"
            "- Local: C:/data/file.csv\n"
            "- HTTP: https://example.com/data.csv\n"
            "- Kaggle: kaggle://dataset/username/dataset-slug\n"
            "- Google Drive: gdrive://file-id\n"
            "- Google Drive Folder: gdrive-folder://folder-id"
        )
        self._create_tooltip(entry, examples)

        # Button frame for all download options
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(side="left", padx=4, pady=4)
        
        ttk.Button(button_frame, text="Browse File", command=self._browse_file).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Load", command=self._open_data_dialog).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Browse Folder", command=self._browse_folder).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Kaggle Search", command=self._search_kaggle).pack(side="left", padx=2)

        main_paned.add(top_section, weight=0)

        # ==================== MIDDLE SECTION: Directory Viewer ====================
        middle_section = ttk.LabelFrame(main_paned, text="Directory Viewer")
        middle_section.pack(fill="both", expand=True, padx=6, pady=6)
        
        # Directory tree and file list in horizontal paned window
        dir_paned = ttk.Panedwindow(middle_section, orient="horizontal")
        dir_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left: Directory tree
        tree_frame = ttk.LabelFrame(dir_paned, text="Folders")
        tree_frame.pack(fill="both", expand=True)
        
        self.dir_tree = ttk.Treeview(tree_frame, show="tree")
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.dir_tree.yview)
        self.dir_tree.configure(yscrollcommand=tree_scroll.set)
        self.dir_tree.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        tree_scroll.pack(side="right", fill="y", padx=2, pady=2)
        
        # Right: File list and info
        right_frame = ttk.Frame(dir_paned)
        right_frame.pack(fill="both", expand=True)
        
        # File list
        file_frame = ttk.LabelFrame(right_frame, text="Files")
        file_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        self.file_list = tk.Listbox(file_frame)
        file_scroll = ttk.Scrollbar(file_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        file_scroll.pack(side="right", fill="y", padx=2, pady=2)
        
        # File info below file list
        info_frame = ttk.LabelFrame(right_frame, text="File Information")
        info_frame.pack(fill="x", padx=2, pady=2)
        
        self.data_info_text = tk.Text(info_frame, width=40, height=6)
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.data_info_text.yview)
        self.data_info_text.configure(yscrollcommand=info_scroll.set)
        self.data_info_text.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        info_scroll.pack(side="right", fill="y", padx=2, pady=2)
        
        dir_paned.add(tree_frame, weight=1)
        dir_paned.add(right_frame, weight=2)
        
        # Bind events
        self.dir_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)
        self.file_list.bind("<Double-1>", self._on_file_double_click)

        main_paned.add(middle_section, weight=1)

        # ==================== BOTTOM SECTION: Data Preview ====================
        bottom_section = ttk.LabelFrame(main_paned, text="Data Preview")
        bottom_section.pack(fill="both", expand=True, padx=6, pady=6)
        
        # The actual preview frame (using your DataPreviewFrame)
        self.preview_frame_data = DataPreviewFrame(bottom_section)
        self.preview_frame_data.pack(fill="both", expand=True, padx=6, pady=6)

        main_paned.add(bottom_section, weight=2)

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1)
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if path:
            self.path_var.set(path)
            self._load_directory_view(os.path.dirname(path))

    def _browse_folder(self):
        """Open folder browser and load directory view"""
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            self.path_var.set(folder_path)
            self._load_directory_view(folder_path)

    def _load_directory_view(self, folder_path):
        """Load directory tree and file list"""
        # Clear existing items
        self.dir_tree.delete(*self.dir_tree.get_children())
        self.file_list.delete(0, tk.END)
        
        # Add root directory
        root_node = self.dir_tree.insert("", "end", text=folder_path, values=[folder_path], open=True)
        self._populate_tree(root_node, folder_path)
        
        # Populate file list for root directory
        self._populate_file_list(folder_path)

    def _populate_tree(self, parent, path):
        """Populate directory tree"""
        try:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    node = self.dir_tree.insert(parent, "end", text=item, values=[full_path])
                    # Add a dummy node to make it expandable
                    self.dir_tree.insert(node, "end", text="dummy")
        except PermissionError:
            pass

    def _populate_file_list(self, folder_path):
        """Populate file list with supported file types"""
        self.file_list.delete(0, tk.END)
        try:
            files = []
            for file in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path):
                    files.append((file, full_path))
            
            # Sort files by name
            files.sort(key=lambda x: x[0].lower())
            
            for file, full_path in files:
                self.file_list.insert(tk.END, file)
                # Store full path as item data (using internal list)
                self.file_list.file_paths = getattr(self.file_list, 'file_paths', {})
                self.file_list.file_paths[self.file_list.size() - 1] = full_path
                
        except PermissionError:
            pass

    def _on_tree_select(self, event):
        """Handle directory tree selection"""
        selection = self.dir_tree.selection()
        if selection:
            folder_path = self.dir_tree.item(selection[0], "values")[0]
            self._populate_file_list(folder_path)
            
            # Expand node if it has dummy child
            children = self.dir_tree.get_children(selection[0])
            if children and self.dir_tree.item(children[0], "text") == "dummy":
                self.dir_tree.delete(children[0])  # Remove dummy
                self._populate_tree(selection[0], folder_path)

    def _on_file_select(self, event):
        """Handle file selection in list"""
        selection = self.file_list.curselection()
        if selection and hasattr(self.file_list, 'file_paths'):
            file_path = self.file_list.file_paths[selection[0]]
            file_info = self._get_file_info(file_path)
            self.data_info_text.delete("1.0", tk.END)
            self.data_info_text.insert("1.0", file_info)
            
            # NEW: Try to preview the data if it's a supported file
            if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                try:
                    df = self._read_file(file_path)
                    if df is not None:
                        self.preview_frame_data.set_dataframe(df)
                    else:
                        self.preview_frame_data.set_text(f"Could not load data from {os.path.basename(file_path)}")
                except Exception as e:
                    self.preview_frame_data.set_text(f"Error reading {os.path.basename(file_path)}: {str(e)}")
            else:
                self.preview_frame_data.set_text(f"Preview not available for {os.path.basename(file_path)}\n\nSupported formats: CSV, Excel (.xlsx, .xls)")

    def _on_file_double_click(self, event):
        """Handle file double-click to load"""
        selection = self.file_list.curselection()
        if selection and hasattr(self.file_list, 'file_paths'):
            file_path = self.file_list.file_paths[selection[0]]
            if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                self.path_var.set(file_path)
                self._open_data_dialog()

    def _read_file(self, path):
        """Helper method to read files for preview"""
        if path.lower().endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(path)
            df = pd.read_excel(xls, sheet_name=0)
            df.__excel_sheets__ = xls.sheet_names
            return df
        elif path.lower().endswith('.csv'):
            df = pd.read_csv(path)
            return df
        else:
            return None
    
    def _get_file_info(self, file_path):
        """Get file information for display"""
        try:
            stat = os.stat(file_path)
            size_kb = stat.st_size / 1024
            file_info = f"File: {os.path.basename(file_path)}\n"
            file_info += f"Path: {file_path}\n"
            file_info += f"Size: {size_kb:.2f} KB\n"
            file_info += f"Modified: {pd.Timestamp.fromtimestamp(stat.st_mtime)}\n"
            file_info += f"Type: {os.path.splitext(file_path)[1]}\n"
            
            # Try to get data info if it's a supported file
            if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                try:
                    df = self._read_file(file_path)
                    if df is not None:
                        file_info += f"\nData Info:\n"
                        file_info += f"Rows: {len(df)}\n"
                        file_info += f"Columns: {len(df.columns)}\n"
                        file_info += f"Columns: {', '.join(df.columns.tolist()[:5])}"
                        if len(df.columns) > 5:
                            file_info += f"... (+{len(df.columns)-5} more)"
                except Exception as e:
                    file_info += f"\nData Info: Error reading file - {str(e)}"
                
            return file_info
        except Exception as e:
            return f"Error getting file info: {str(e)}"

    def _search_kaggle(self):
        """Open Kaggle search dialog"""
        search_window = tk.Toplevel(self.root)
        search_window.title("Kaggle Search")
        search_window.geometry("500x300")
        
        ttk.Label(search_window, text="Search Kaggle Datasets:").pack(pady=10)
        
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_window, textvariable=search_var, width=50)
        search_entry.pack(pady=5)
        
        results_frame = ttk.Frame(search_window)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        listbox = tk.Listbox(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def perform_search():
            query = search_var.get().strip()
            if not query:
                return
            
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                datasets = api.dataset_list(search=query)
                listbox.delete(0, "end")
                
                for dataset in datasets[:10]:  # Show first 10 results
                    listbox.insert("end", f"{dataset.ref}")
                    
            except ImportError:
                messagebox.showerror("Error", "Kaggle API not installed. Run: pip install kaggle")
            except Exception as e:
                messagebox.showerror("Error", f"Search failed: {str(e)}")
        
        def on_select(event):
            selection = listbox.curselection()
            if selection:
                dataset_ref = listbox.get(selection[0])
                self.path_var.set(f"kaggle://dataset/{dataset_ref}")
                search_window.destroy()
        
        ttk.Button(search_window, text="Search", command=perform_search).pack(pady=5)
        listbox.bind("<Double-1>", on_select)

    def _open_data_dialog(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showerror("No path", "Enter a file path or URL first.")
            return
        
        # Show progress bar
        self.progress_var.set(0)
        self.progress_bar.grid(row=2, column=0, sticky="ew")
        self.status.config(text="Loading data...")
        
        # Use after() for non-blocking loading
        self.root.after(100, lambda: self._load_data_with_progress(path))

    def _load_data_with_progress(self, path):
        """Load data without threading"""
        try:
            def progress_callback(progress, message):
                self.progress_var.set(progress)
                self.status.config(text=message)
                self.root.update_idletasks()
            
            output_dir = getattr(self, 'OUTPUT_DIR', OUTPUT_DIR)
            
            df, saved = load_file_or_url(path, output_dir, progress_callback)
            
            self._on_data_loaded(df, saved)
            
        except Exception as e:
            self._on_load_error(str(e))

    def _on_data_loaded(self, df, saved):
        """Handle successful data load"""
        self.progress_bar.grid_remove()
        
        if df is None:
            messagebox.showinfo("No data", "File loaded but no readable data found. Check folder browser for individual files.")
            return
            
        self.df_original = df.copy()
        self.filepath = saved
        self.preprocessor = Preprocessor(self.df_original)
        self.preview_df = self.df_original.copy()
        self.df_processed = self.df_original.copy()
        self._update_data_info()
        # Use the preview frame from data tab
        for widget in self.tab_data.winfo_children():
            if isinstance(widget, ttk.Panedwindow):
                for child in widget.winfo_children():
                    if hasattr(child, 'set_dataframe'):
                        child.set_dataframe(self.preview_df)
                        break
        self.status.config(text=f"Loaded: {saved}")
        self.notebook.select(self.tab_preproc)
        self._populate_preproc_columns()

    def _on_load_error(self, error_msg):
        """Handle data load error"""
        self.progress_bar.grid_remove()
        messagebox.showerror("Load failed", error_msg)
        self.status.config(text="Load failed")

    def _update_data_info(self):
        if self.df_original is None:
            return
        txt = f"Saved path: {self.filepath}\n"
        txt += get_basic_stats(self.df_original) + "\n\n"
        txt += get_inspection_html(self.df_original)
        self.data_info_text.delete("1.0", "end")
        self.data_info_text.insert("1.0", txt)

    def _preview_head(self):
        if self.preview_df is None:
            return
        # Use the preview frame from data tab
        for widget in self.tab_data.winfo_children():
            if isinstance(widget, ttk.Panedwindow):
                for child in widget.winfo_children():
                    if hasattr(child, 'set_dataframe'):
                        child.set_dataframe(self.preview_df.head(20))
                        break

    def _inspect_data(self):
        if self.df_original is None:
            return
        txt_df = self.df_original.describe(include="all").fillna("").to_string()
        # Use the preview frame from data tab
        for widget in self.tab_data.winfo_children():
            if isinstance(widget, ttk.Panedwindow):
                for child in widget.winfo_children():
                    if hasattr(child, 'set_text'):
                        child.set_text(txt_df)
                        break

    # ----------------------------
    # PREPROCESS TAB (horizontal paned)
    # ----------------------------
    def _build_preproc_tab(self):
            # Main horizontal layout
            main_paned = ttk.Panedwindow(self.tab_preproc, orient="horizontal")
            main_paned.pack(fill="both", expand=True)

            # ==================== LEFT SECTION: Target & Global Settings ====================
            left_section = ttk.Frame(main_paned)
            left_section.pack(fill="both", expand=True, padx=6, pady=6)

            # Target and Features selection
            selection_frame = ttk.LabelFrame(left_section, text="Target & Features Selection")
            selection_frame.pack(fill="x", padx=6, pady=6)
            
            # Target selection
            ttk.Label(selection_frame, text="Target Column (y):").pack(anchor="w", padx=6, pady=(6,0))
            self.target_combo = ttk.Combobox(selection_frame, state="readonly", values=[], width=30)
            self.target_combo.pack(padx=6, pady=4, fill="x")
            
            # Features selection
            ttk.Label(selection_frame, text="Feature Columns (x):").pack(anchor="w", padx=6, pady=(10,0))
            
            self.features_var = tk.StringVar(value="No features selected")
            features_display = ttk.Entry(selection_frame, textvariable=self.features_var, state="readonly", width=30)
            features_display.pack(padx=6, pady=4, fill="x")
            
            # Feature selection button
            ttk.Button(selection_frame, text="Choose Features...", command=self._open_features_dialog).pack(padx=6, pady=4)

            # ==================== GLOBAL TRANSFORMATIONS ====================
            global_frame = ttk.LabelFrame(left_section, text="Global Transformations")
            global_frame.pack(fill="x", padx=6, pady=6)
            
            row = 0
            
            # Missing value removal
            self.remove_all_missing_var = tk.BooleanVar(value=False)
            remove_missing_cb = ttk.Checkbutton(global_frame, 
                                            text="Remove rows with ANY missing values", 
                                            variable=self.remove_all_missing_var,
                                            command=self._on_preproc_change)
            remove_missing_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4)
            row += 1
            
            # Encoding method
            ttk.Label(global_frame, text="Categorical Encoding:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
            self.encode_method = tk.StringVar(value="none")
            encoding_combo = ttk.Combobox(global_frame, textvariable=self.encode_method, 
                                        values=["none", "onehot", "label"], 
                                        state="readonly", width=12)
            encoding_combo.grid(row=row, column=1, padx=6, pady=4)
            encoding_combo.bind('<<ComboboxSelected>>', self._on_preproc_change)
            row += 1
            
            # Data transformations
            self.transpose_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(global_frame, text="Transpose data", variable=self.transpose_var, 
                        command=self._on_preproc_change).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            row += 1
            
            self.remove_neg_var = tk.BooleanVar(value=False)
            remove_neg_cb = ttk.Checkbutton(global_frame, text="Replace negative values", variable=self.remove_neg_var,
                        command=self._on_preproc_change)
            remove_neg_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            row += 1
            
            ttk.Label(global_frame, text="Replace with:").grid(row=row, column=0, sticky="w", padx=6, pady=2)
            self.replace_neg_value = tk.StringVar(value="0")
            ttk.Entry(global_frame, textvariable=self.replace_neg_value, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=2)

            self.remove_rare_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(global_frame, text="Remove rare target classes (n < 2)", 
                                variable=self.remove_rare_var,
                                command=self._on_preproc_change).grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=2)  
                              
            global_frame.columnconfigure(0, weight=1)
            global_frame.columnconfigure(1, weight=0)


            # === BUTTONS SECTION ===
            # Standard Apply Button
            ttk.Button(left_section, text="Apply Preprocessing", command=self._apply_preprocessing).pack(fill='x', padx=6, pady=(10, 2))
            
            # 🟢 NEW: Advanced Tools Button (PCA/Scripting)
            btn_style = ttk.Style()
            btn_style.configure("Bold.TButton", font=('Sans','10','bold'))
            ttk.Button(left_section, text="✨ Advanced Tools (PCA / Scripting)", style="Bold.TButton",
                    command=self._open_enhanced_preprocessor).pack(fill='x', padx=6, pady=2)

            main_paned.add(left_section, weight=1)

            # ==================== MIDDLE SECTION: Per-Column Settings ====================
            middle_section = ttk.LabelFrame(main_paned, text="Per-Column Settings")
            middle_section.pack(fill="both", expand=True, padx=6, pady=6)
            
            canvas_frame = ttk.Frame(middle_section)
            canvas_frame.pack(fill="both", expand=True, padx=2, pady=2)
            
            self.col_controls_canvas = tk.Canvas(canvas_frame, height=300)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.col_controls_canvas.yview)
            self.col_controls_scrollable = ttk.Frame(self.col_controls_canvas)
            
            self.col_controls_scrollable.bind(
                "<Configure>",
                lambda e: self.col_controls_canvas.configure(scrollregion=self.col_controls_canvas.bbox("all"))
            )
            
            self.col_controls_canvas.create_window((0, 0), window=self.col_controls_scrollable, anchor="nw")
            self.col_controls_canvas.configure(yscrollcommand=scrollbar.set)
            
            self.col_controls_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            headers_frame = ttk.Frame(self.col_controls_scrollable)
            headers_frame.pack(fill="x", padx=4, pady=2)
            ttk.Label(headers_frame, text="Column", width=15).pack(side="left", padx=2)
            ttk.Label(headers_frame, text="Advanced", width=10).pack(side="left", padx=2)

            main_paned.add(middle_section, weight=2)

            # ==================== RIGHT SECTION: Preview ====================
            right_section = ttk.LabelFrame(main_paned, text="Preprocessing Preview")
            right_section.pack(fill="both", expand=True, padx=6, pady=6)
            
            self.preview_frame_preproc = DataPreviewFrame(right_section)
            self.preview_frame_preproc.pack(fill="both", expand=True, padx=6, pady=6)

            main_paned.add(right_section, weight=2)

    def _open_features_dialog(self):
        """Open dialog for feature selection"""
        if not hasattr(self, 'df_original') or self.df_original is None:
            messagebox.showinfo("No Data", "Load data first.")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Features")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Select all checkbox
        select_all_var = tk.BooleanVar(value=True)
        
        def toggle_all():
            for var in self.feature_vars.values():
                var.set(select_all_var.get())
            self._update_features_display()
        
        ttk.Checkbutton(dialog, text="Select All", variable=select_all_var, 
                    command=toggle_all).pack(anchor="w", padx=10, pady=5)
        
        # Scrollable frame for checkboxes
        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(frame, height=400)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create checkboxes for each column
        for col in self.df_original.columns:
            var = self.feature_vars.get(col, tk.BooleanVar(value=True))
            self.feature_vars[col] = var
            
            cb = ttk.Checkbutton(scrollable_frame, text=col, variable=var)
            cb.pack(anchor="w", padx=5, pady=2)
        
        # OK button
        def on_ok():
            self._update_features_display()
            self._update_selected_features_display()
            self._on_preproc_change()
            dialog.destroy()
            
        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        
        # Update display initially
        self._update_features_display()

    def _update_features_display(self):
        """Update the features display text"""
        if not hasattr(self, 'feature_vars'):
            return
            
        selected = [col for col, var in self.feature_vars.items() if var.get()]
        if selected:
            text = f"{len(selected)} features selected"
            if len(selected) <= 3:
                text = ", ".join(selected)
            else:
                text = f"{len(selected)} features: {', '.join(selected[:3])}..."
            self.features_var.set(text)
        else:
            self.features_var.set("No features selected")

    def _open_column_editor(self, column):
        """Open the advanced editor for a given column"""
        if self.df_original is None:
            return

        # Extract all tk variables for this column
        column_vars = self.per_col_widgets[column]

        # Convert tk.StringVar → raw string values
        settings_dict = {
            key: (var.get() if hasattr(var, "get") else var)
            for key, var in column_vars.items()
            if key != "summary_label"
        }

        # Callback from editor → update our variables + update preview
        def on_settings_change(updated_column, new_settings):
            # Update all tk variables
            col_vars = self.per_col_widgets[updated_column]
            for key, value in new_settings.items():
                if key in col_vars:
                    col_vars[key].set(value)

            # Refresh the small summary row
            self._update_selected_features_display()

            # Recompute preview
            self._on_preproc_change()

        # OPEN THE EDITOR
        AdvancedEditor(
            self.root,                # parent window
            self.preprocessor,        # preprocessor instance
            column,                   # column name
            settings_dict,            # dict of settings
            on_settings_change        # callback
        )
        
    def _show_mapping_summary(self, column, mappings):
        """Show a summary of applied mappings in a popup"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title(f"Mapping Summary - {column}")
        summary_window.geometry("500x400")
        
        text_widget = tk.Text(summary_window, wrap="word")
        scrollbar = ttk.Scrollbar(summary_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y", padx=5, pady=5)
        
        # Create summary
        summary = f"Mapping Summary for column: {column}\n"
        summary += "=" * 50 + "\n\n"
        
        if mappings:
            # Group by target labels
            label_groups = {}
            for mapping in mappings:
                if ":" in mapping:
                    original, new_label = mapping.split(":", 1)
                    if new_label not in label_groups:
                        label_groups[new_label] = []
                    label_groups[new_label].append(original.strip())
            
            for label, values in label_groups.items():
                summary += f"→ '{label}':\n"
                for value in values[:5]:
                    summary += f"   - {value}\n"
                if len(values) > 5:
                    summary += f"   ... and {len(values) - 5} more values\n"
                summary += "\n"
        else:
            summary += "No mappings applied yet.\n"
        
        text_widget.insert("1.0", summary)
        text_widget.config(state="disabled")
        
        ttk.Button(summary_window, text="Close", 
                command=summary_window.destroy).pack(pady=10)
    
    def _populate_preproc_columns(self):
        # Clear existing widgets from the scrollable area
        for widget in self.col_controls_scrollable.winfo_children():
            if isinstance(widget, ttk.Frame) and widget != self.col_controls_scrollable.winfo_children()[0]:
                widget.destroy()

        if self.df_original is None:
            return
            
        columns = list(self.df_original.columns)
        
        # Set target combo
        self.target_combo.config(values=columns)
        if columns:
            self.target_combo.set(columns[0])
        
        # Initialize feature selection
        self.feature_vars = {}
        for col in columns:
            self.feature_vars[col] = tk.BooleanVar(value=True)
        self._update_features_display()
        
        # Create per-column controls
        self.per_col_widgets = {}
        
        for col in columns:
            self.per_col_widgets[col] = {
                "strategy": tk.StringVar(value="none"),
                "fill_value": tk.StringVar(value=""),
                "low": tk.StringVar(value=""),
                "high": tk.StringVar(value=""),
                "onehot": tk.BooleanVar(value=False),
                "label": tk.BooleanVar(value=False),
                "custom_mapping": tk.StringVar(value=""),
                "bins": tk.StringVar(value=""),
                "bin_labels": tk.StringVar(value=""),
                "pattern_groups": tk.StringVar(value=""),
                "remove_values": tk.StringVar(value=""),
                "remove_mode": tk.StringVar(value="equals"),
                "clamp_min": tk.StringVar(value=""),
                "clamp_min_to": tk.StringVar(value=""),
                "clamp_max": tk.StringVar(value=""),
                "clamp_max_to": tk.StringVar(value=""),
                "percent_rules": tk.StringVar(value=""),
                "strip_substring": tk.StringVar(value=""),
                "dtype_conversion": tk.StringVar(value="")
            }

        # Update the display to show current selection
        self._update_selected_features_display()

        self.preprocessor = Preprocessor(self.df_original)

    def _update_selected_features_display(self):
        """Update the display to show only selected features"""
        # Clear existing widgets
        for widget in self.col_controls_scrollable.winfo_children():
            if isinstance(widget, ttk.Frame) and widget != self.col_controls_scrollable.winfo_children()[0]:
                widget.destroy()

        if not hasattr(self, 'feature_vars'):
            return
            
        # Get selected features
        selected_features = [col for col, var in self.feature_vars.items() if var.get()]
        
        if not selected_features:
            # Show message when no features selected
            empty_frame = ttk.Frame(self.col_controls_scrollable)
            empty_frame.pack(fill="x", padx=4, pady=10)
            ttk.Label(empty_frame, text="No features selected. Click 'Choose Features...' to select features.", 
                    foreground="gray").pack()
            return
        
        # Create rows only for selected features
        for col in selected_features:
            frame = ttk.Frame(self.col_controls_scrollable)
            frame.pack(fill="x", padx=4, pady=2)
            
            # Column name
            ttk.Label(frame, text=col, width=20, anchor="w").pack(side="left", padx=2)
            
            # Advanced editor button
            edit_btn = ttk.Button(frame, text="...", width=3,
                                command=lambda c=col: self._open_column_editor(c))
            edit_btn.pack(side="left", padx=2)
            
            # Current settings summary
            settings_summary = self._get_settings_summary(col)
            summary_label = ttk.Label(frame, text=settings_summary, width=40, anchor="w")
            summary_label.pack(side="left", padx=2)
            
            # Store the summary label for updates
            if col not in self.per_col_widgets:
                self.per_col_widgets[col] = {}
            self.per_col_widgets[col]["summary_label"] = summary_label

    def _get_settings_summary(self, column):
        """Get a summary of current settings for a column"""
        if column not in self.per_col_widgets:
            return "No settings"
        
        settings = self.per_col_widgets[column]
        summary_parts = []
        
        # Check basic settings
        if settings["strategy"].get() != "none":
            summary_parts.append(f"Missing:{settings['strategy'].get()}")
        
        if settings["low"].get():
            summary_parts.append(f"Min:{settings['low'].get()}")
        
        if settings["high"].get():
            summary_parts.append(f"Max:{settings['high'].get()}")
        
        if settings["onehot"].get():
            summary_parts.append("OneHot")
        
        if settings["label"].get():
            summary_parts.append("Label")
        
        if settings["custom_mapping"].get():
            summary_parts.append("CustomMap")
        
        if settings["bins"].get():
            summary_parts.append("Binning")
        
        if settings["pattern_groups"].get():
            summary_parts.append("Patterns")
            
        if settings.get("dtype_conversion", tk.StringVar(value="")).get():
            dtype = settings["dtype_conversion"].get()
            summary_parts.append(f"→{dtype}")
        
        # Add the new settings to summary
        if settings.get("remove_values", tk.StringVar(value="")).get():
            summary_parts.append("RowFilter")
        
        if settings.get("clamp_min", tk.StringVar(value="")).get() or settings.get("clamp_max", tk.StringVar(value="")).get():
            summary_parts.append("Clamp")
        
        if settings.get("percent_rules", tk.StringVar(value="")).get():
            summary_parts.append("PercentRules")
            
        if settings.get("strip_substring", tk.StringVar(value="")).get():
            summary_parts.append("StripSubstring")

        return ", ".join(summary_parts) if summary_parts else "Default settings"

    def _select_all_features(self):
        """Select all features in the feature dialog"""
        if hasattr(self, 'feature_vars'):
            for var in self.feature_vars.values():
                var.set(True)
            self._update_features_display()
            self._on_preproc_change()

    def _on_preproc_change(self, *_):
        """Handle preprocessing changes and update preview"""
        if self.df_original is None:
            return
            
        settings = {
            "transpose": self.transpose_var.get(),
            "encode_method": self.encode_method.get(),
            "replace_negatives": self.remove_neg_var.get(),
            "replace_neg_value": self.replace_neg_value.get(),
            "remove_rare_target": self.remove_rare_var.get(),
            "remove_all_missing": self.remove_all_missing_var.get()
        }
        
        # Get only settings for selected features
        per_col = {}
        selected_features = [c for c, v in getattr(self, "feature_vars", {}).items() if v.get()]
        
        for col in selected_features:
            if col in self.per_col_widgets:
                widgets = self.per_col_widgets[col]
                per_col[col] = {
                    "strategy": widgets["strategy"].get(),
                    "fill_value": widgets["fill_value"].get(),
                    "low": widgets["low"].get(),
                    "high": widgets["high"].get(),
                    "onehot": widgets["onehot"].get(),
                    "label": widgets["label"].get(),
                    "custom_mapping": widgets["custom_mapping"].get(),
                    "bins": widgets["bins"].get(),
                    "bin_labels": widgets["bin_labels"].get(),
                    "pattern_groups": widgets["pattern_groups"].get(),
                    "remove_values": widgets.get("remove_values", tk.StringVar(value="")).get(),
                    "remove_mode": widgets.get("remove_mode", tk.StringVar(value="equals")).get(),
                    "clamp_min": widgets.get("clamp_min", tk.StringVar(value="")).get(),
                    "clamp_min_to": widgets.get("clamp_min_to", tk.StringVar(value="")).get(),
                    "clamp_max": widgets.get("clamp_max", tk.StringVar(value="")).get(),
                    "clamp_max_to": widgets.get("clamp_max_to", tk.StringVar(value="")).get(),
                    "percent_rules": widgets.get("percent_rules", tk.StringVar(value="")).get(),
                    "strip_substring": widgets.get("strip_substring", tk.StringVar(value="")).get(),
                    "dtype_conversion": widgets.get("dtype_conversion", tk.StringVar(value="")).get()
                }
                
        # Debug call
        self._debug_settings(per_col, settings)
        
        target = self.target_combo.get() if self.target_combo.get() else None

        try:
            # Use the new AdvancedPreprocessor
            preview = self.preprocessor.apply_advanced_transformations(
                column_settings=per_col, 
                global_settings=settings,
                selected_features=selected_features, 
                target=target
            )
            
            # Update selected_features to reflect actual column changes
            actual_features = [col for col in preview.columns if col != target]
            
            self.preview_df = preview
            self.preview_frame_preproc.set_dataframe(self.preview_df)
            
            # Show transformation summary
            self._show_transformation_summary()
            
            # Update status with transformation info
            logs = self.preprocessor.get_transformations_log()
            if logs:
                self.status.config(text=f"Preview: {len(logs)} transformations applied")
            else:
                self.status.config(text="Preview updated")
                
        except Exception as e:
            self.preview_frame_preproc.set_text(f"Error in preprocessing: {str(e)}")
            self.status.config(text="Error in preprocessing")

    def _apply_preprocessing(self):
        """Apply the preprocessing changes and ensure proper data types"""
        if self.preview_df is None:
            messagebox.showinfo("Nothing to apply", "No preview available.")
            return
        
        try:
            # Make a deep copy of the preview data
            self.df_processed = self.preview_df.copy()
            
            # Auto-convert object columns to numeric where possible
            self.df_processed = self._auto_convert_to_numeric(self.df_processed)
            
            self.status.config(text="Preprocessing applied. Data types optimized for ML.")
            
            # Show data type summary
            numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            
            summary = f"Applied preprocessing!\n"
            summary += f"Numeric columns: {len(numeric_cols)}\n"
            summary += f"Categorical columns: {len(categorical_cols)}\n"
            if numeric_cols:
                summary += f"Numeric: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}"
            
            messagebox.showinfo("Applied", summary)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preprocessing: {str(e)}")

    def _auto_convert_to_numeric(self, df):
        """Automatically convert object columns to numeric where possible"""
        df = df.copy()
        
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # Check if conversion was successful (not all NaN)
            if not numeric_series.isna().all():
                # Check if we have a reasonable conversion rate (at least 80% success)
                conversion_rate = 1 - (numeric_series.isna().sum() / len(df))
                if conversion_rate >= 0.8:
                    df[col] = numeric_series
                    print(f"Auto-converted {col} to numeric (success rate: {conversion_rate:.1%})")
        
        return df

    # ----------------------------
    # ML TAB
    # ----------------------------
    def _build_ml_tab(self):
        """Build a comprehensive ML tab with Feature Engineering options."""
        # Clear existing tab
        for widget in self.tab_ml.winfo_children():
            widget.destroy()

        # Define model_lists
        self.model_lists = {
            "classification": [" ","lr", "dt", "rf", "xgboost", "lightgbm", "svm", "knn", "nb", "ada", "gbc"],
            "regression": [" ","lr", "lasso", "ridge", "dt", "rf", "xgboost", "lightgbm", "svm", "knn", "et"],
            "clustering": [" ","kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"],
            "anomaly": [" ","iforest", "cluster", "cof", "histogram", "knn", "lof", "svm", "pca"]
        }
        
        # Main paned window
        main_paned = ttk.Panedwindow(self.tab_ml, orient="horizontal")
        main_paned.pack(fill="both", expand=True)
        
        # ==================== LEFT: Controls ====================
        left_frame = ttk.LabelFrame(main_paned, text="Model Controls", padding=10)
        left_frame.pack(fill="both", expand=True)
        
        # Target selection
        ttk.Label(left_frame, text="Target Column:").pack(anchor="w", pady=(0,5))
        df_current = getattr(self, "df_processed", None) or getattr(self, "df_original", None)
        columns = list(df_current.columns) if df_current is not None else []
        self.ml_target_combo = ttk.Combobox(left_frame, values=columns, state="readonly")
        self.ml_target_combo.pack(fill="x", pady=(0,10))
        if columns:
            self.ml_target_combo.set(columns[0])
        
        # Features selection
        ttk.Label(left_frame, text="Feature Columns:").pack(anchor="w", pady=(0,5))
        feature_container = ttk.Frame(left_frame)
        feature_container.pack(fill="both", expand=True)
        
        feature_canvas = tk.Canvas(feature_container, height=150)
        scrollbar = ttk.Scrollbar(feature_container, orient="vertical", command=feature_canvas.yview)
        self.scrollable_feature_frame = ttk.Frame(feature_canvas)
        
        self.scrollable_feature_frame.bind("<Configure>", 
            lambda e: feature_canvas.configure(scrollregion=feature_canvas.bbox("all")))
        
        feature_canvas.create_window((0, 0), window=self.scrollable_feature_frame, anchor="nw")
        feature_canvas.configure(yscrollcommand=scrollbar.set)
        
        feature_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.ml_feature_vars = {}
        self._populate_ml_features()
        
        ttk.Button(left_frame, text="Refresh Columns", command=self._refresh_ml_columns).pack(fill="x", pady=10)
        
        main_paned.add(left_frame, weight=1)
        
        # ==================== CENTER: Configuration ====================
        center_container = ttk.Frame(main_paned)
        center_container.pack(fill="both", expand=True)
        
        config_canvas = tk.Canvas(center_container)
        config_scrollbar = ttk.Scrollbar(center_container, orient="vertical", command=config_canvas.yview)
        self.scrollable_config_frame = ttk.Frame(config_canvas)
        
        self.scrollable_config_frame.bind("<Configure>",
            lambda e: config_canvas.configure(scrollregion=config_canvas.bbox("all")))
        
        config_canvas.create_window((0, 0), window=self.scrollable_config_frame, anchor="nw")
        config_canvas.configure(yscrollcommand=config_scrollbar.set)
        
        config_canvas.pack(side="left", fill="both", expand=True)
        config_scrollbar.pack(side="right", fill="y")
        
        config_frame = ttk.LabelFrame(self.scrollable_config_frame, text="Model Configuration", padding=10)
        config_frame.pack(fill="x", padx=5, pady=5)

        # Problem & Model
        problem_row = ttk.Frame(config_frame)
        problem_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(problem_row, text="Problem Type:", width=15).pack(side="left", padx=2)
        self.problem_type = tk.StringVar(value="auto")
        problem_combo = ttk.Combobox(problem_row, textvariable=self.problem_type, 
                                    values=["auto", "classification", "regression", "clustering", "anomaly"],
                                    state="readonly", width=15)
        problem_combo.pack(side="left", padx=2, fill="x", expand=True)
        problem_combo.bind("<<ComboboxSelected>>", self._on_problem_type_changed)
        
        model_row = ttk.Frame(config_frame)
        model_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(model_row, text="Select Model:", width=15).pack(side="left", padx=2)
        self.model_selection = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_selection, 
                                    state="readonly", width=15)
        self.model_combo.pack(side="left", padx=2, fill="x", expand=True)
        
        self.experiment_type = tk.StringVar(value="compare")

        # Hyperparams button
        hyper_frame = ttk.Frame(config_frame)
        hyper_frame.pack(fill="x", padx=5, pady=2)
        self.hyperparams_button = ttk.Button(
            hyper_frame, text="Configure Hyperparameters", command=self._open_hyperparameter_editor
        )
        self.hyperparams_button.pack(anchor="w", padx=5, pady=2)
        if hasattr(self, 'model_combo'):
            self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selection_changed)
        self.model_hyperparameters = {}        

        # --- FEATURE ENGINEERING (NEW) ---
        fe_frame = ttk.LabelFrame(config_frame, text="Curve Fitting & Polynomials", padding=5)
        fe_frame.pack(fill="x", padx=5, pady=10)
        
        # Polynomial Features Checkbox
        self.poly_features = tk.BooleanVar(value=False)
        ttk.Checkbutton(fe_frame, text="Use Polynomial Features (Curve Fitting)", 
                        variable=self.poly_features).pack(anchor="w", padx=5, pady=2)
        
        # Degree Spinner
        deg_row = ttk.Frame(fe_frame)
        deg_row.pack(fill="x", padx=20, pady=2)
        ttk.Label(deg_row, text="Degree (Power):").pack(side="left")
        self.poly_degree = tk.IntVar(value=2)
        ttk.Spinbox(deg_row, from_=2, to=5, textvariable=self.poly_degree, width=3).pack(side="left", padx=5)
        
        # Interaction Features
        self.interaction_features = tk.BooleanVar(value=False)
        ttk.Checkbutton(fe_frame, text="Include Interactions (A*B)", 
                       variable=self.interaction_features).pack(anchor="w", padx=5, pady=2)

        # --- Validation Settings (Hidden by default) ---
        self.show_validation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Show Advanced Validation (Split/CV)", 
                    variable=self.show_validation_var, 
                    command=self._toggle_validation_frame).pack(anchor="w", padx=5, pady=10)
        
        self.validation_frame = ttk.LabelFrame(config_frame, text="Validation Settings", padding=5)
        
        split_row = ttk.Frame(self.validation_frame)
        split_row.pack(fill="x", pady=2)
        ttk.Label(split_row, text="Test Size:").pack(side="left", padx=2)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_slider = ttk.Scale(split_row, from_=0.1, to=0.5, variable=self.test_size_var, orient="horizontal")
        test_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.test_size_label = ttk.Label(split_row, text="20%")
        self.test_size_label.pack(side="left", padx=5)
        
        def update_test_size_label(*args):
            self.test_size_label.config(text=f"{int(self.test_size_var.get()*100)}%")
        self.test_size_var.trace('w', update_test_size_label)
        
        cv_row = ttk.Frame(self.validation_frame)
        cv_row.pack(fill="x", pady=2)
        ttk.Label(cv_row, text="CV Folds:", width=15).pack(side="left", padx=2)
        self.cv_folds = tk.IntVar(value=5)
        ttk.Spinbox(cv_row, from_=2, to=10, textvariable=self.cv_folds, width=5).pack(side="left", padx=2)

        seed_row = ttk.Frame(self.validation_frame)
        seed_row.pack(fill="x", pady=2)
        ttk.Label(seed_row, text="Seed:", width=15).pack(side="left", padx=2)
        self.session_id = tk.IntVar(value=42)
        self.random_state_var = tk.IntVar(value=42)
        ttk.Entry(seed_row, textvariable=self.session_id, width=8).pack(side="left", padx=2)

        # Other options
        self.tune_model = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Tune Hyperparameters", variable=self.tune_model).pack(anchor="w", padx=5, pady=2)
        
        self.finalize_model = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Finalize Model", variable=self.finalize_model).pack(anchor="w", padx=5, pady=2)
        
        self.use_nan_test = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Use NaN as test set", variable=self.use_nan_test).pack(anchor="w", padx=5, pady=2)
        
        # Apply Model Button
        self.apply_model_btn = ttk.Button(config_frame, text="Run Training", command=self._run_pycaret_training)
        self.apply_model_btn.pack(fill="x", padx=5, pady=10)
        
        main_paned.add(center_container, weight=2)
        
        # ==================== RIGHT: Results & Logs ====================
        right_paned = ttk.Panedwindow(main_paned, orient="vertical")
        right_paned.pack(fill="both", expand=True)
        
        results_frame = ttk.LabelFrame(right_paned, text="Quick Results", padding=10)
        results_frame.pack(fill="both", expand=True)
        
        self.results_text = tk.Text(results_frame, wrap="word", height=15)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        logs_frame = ttk.LabelFrame(right_paned, text="Training Logs", padding=10)
        logs_frame.pack(fill="both", expand=True)
        
        self.ml_logs_text = tk.Text(logs_frame, height=10, wrap="word")
        logs_scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.ml_logs_text.yview)
        self.ml_logs_text.configure(yscrollcommand=logs_scrollbar.set)
        self.ml_logs_text.pack(side="left", fill="both", expand=True)
        logs_scrollbar.pack(side="right", fill="y")
        
        ttk.Button(logs_frame, text="Clear Logs", command=self._clear_ml_logs).pack(anchor="e", pady=5)
        
        right_paned.add(results_frame, weight=1)
        right_paned.add(logs_frame, weight=1)
        main_paned.add(right_paned, weight=2)
        
        self._on_problem_type_changed()
        self._update_ml_preview()

    def _on_model_selection_changed(self, event=None):
        """Enable hyperparameter button when a specific model is selected"""
        model_name = self.model_selection.get().strip()
        if model_name and model_name != " ":
            self.hyperparams_button.config(state="normal")
        else:
            self.hyperparams_button.config(state="disabled")

    def _open_hyperparameter_editor(self):
        """Open hyperparameter configuration dialog"""
        model_name = self.model_selection.get().strip()
        if not model_name:
            return
            
        current_params = self.model_hyperparameters.get(model_name, {})
        
        def on_parameters_save(parameters):
            self.model_hyperparameters[model_name] = parameters
            self._log_ml_message(f"✅ Hyperparameters saved for {model_name}: {parameters}")
        
        from hyperparameter_editor import HyperparameterEditor
        HyperparameterEditor(self.root, model_name, current_params, on_parameters_save)

    def _on_problem_type_changed(self, event=None):
        """Update model selection when problem type changes"""
        problem_type = self.problem_type.get()
        if hasattr(self, 'model_lists') and problem_type in self.model_lists:
            models = self.model_lists[problem_type]
            if hasattr(self, 'model_combo'):
                self.model_combo.config(values=models)
                if models:
                    self.model_combo.set(models[0])

    def _on_experiment_type_changed(self, event=None):
        """Enable/disable model selection based on experiment type"""
        if self.experiment_type.get() == "single":
            self.model_combo.config(state="readonly")
        else:
            self.model_combo.config(state="disabled")

    def _populate_ml_features(self):
        """Populate feature checkboxes (Updated for Empty Target support)"""
        for widget in self.scrollable_feature_frame.winfo_children():
            widget.destroy()
        
        self.ml_feature_vars = {}
        
        # Get Data
        df_processed = getattr(self, "df_processed", None)
        df_original = getattr(self, "df_original", None)
        
        if df_processed is not None:
            df_current = df_processed
        elif df_original is not None:
            df_current = df_original
        else:
            return
            
        columns = list(df_current.columns)
        
        # Get current target
        target = getattr(self, 'ml_target_combo', tk.StringVar()).get()
        
        for col in columns:
            # ONLY skip this column if it matches the target exactly.
            # If target is " " (Space), then 'Price' != ' ', so Price appears!
            if col == target:
                continue
                
            var = tk.BooleanVar(value=True)
            self.ml_feature_vars[col] = var
            
            cb = ttk.Checkbutton(self.scrollable_feature_frame, text=col, variable=var)
            cb.pack(anchor="w", padx=5, pady=1)
        
    def _log_ml_message(self, message):
        """Add message to ML logs with timestamp"""
        if hasattr(self, 'ml_logs_text'):
            timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
            self.ml_logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.ml_logs_text.see(tk.END)
            self.ml_logs_text.update()

    def _clear_ml_logs(self):
        """Clear ML logs"""
        if hasattr(self, 'ml_logs_text'):
            self.ml_logs_text.delete(1.0, tk.END)

    def _refresh_ml_columns(self):
        """Refresh columns in ML tab with persistent Empty Option for Clustering."""
        # Safe dataframe retrieval
        df_processed = getattr(self, "df_processed", None)
        df_original = getattr(self, "df_original", None)
        
        if df_processed is not None:
            df_current = df_processed
        elif df_original is not None:
            df_current = df_original
        else:
            # No data available
            if hasattr(self, 'ml_target_combo'):
                self.ml_target_combo.config(values=[])
                self.ml_target_combo.set('')
            return
        
        columns = list(df_current.columns)
        
        # --- FIX: ADD EMPTY OPTION & RESPECT SELECTION ---
        if hasattr(self, 'ml_target_combo'):
            # 1. Set values with the empty option at the start
            # " " (space) is safer than "" (empty string) for some OS visualizers
            empty_option = " " 
            self.ml_target_combo.config(values=[empty_option] + columns)
            
            # 2. Check current selection
            current_selection = self.ml_target_combo.get()
            
            # 3. Logic: Only force a default if the current selection is INVALID
            # If it's " " (empty), we LEAVE IT ALONE.
            # If it's "Price" (valid), we LEAVE IT ALONE.
            # If it's "" (truly blank on first load), we default to the last column.
            if current_selection == empty_option:
                pass # Do nothing, user wants it empty
            elif current_selection in columns:
                pass # Do nothing, user has a valid column
            else:
                # Default behavior: Pick the last column (usually the target)
                self.ml_target_combo.set(columns[-1])
        
        # Update features list based on the (now stable) target
        self._populate_ml_features()
        self._update_ml_preview()

    def _update_ml_preview(self):
        """Update ML preview with selected features"""
        # Safe dataframe retrieval
        df_processed = getattr(self, "df_processed", None)
        df_original = getattr(self, "df_original", None)
        
        # FIX: Proper None checking
        if df_processed is not None:
            df_current = df_processed
        elif df_original is not None:
            df_current = df_original
        else:
            # No data available
            if hasattr(self, 'ml_preview_frame'):
                self.ml_preview_frame.set_text("No data loaded")
            return
        
        if hasattr(self, 'ml_preview_frame'):
            # Get selected features
            selected_features = []
            if hasattr(self, 'ml_feature_vars'):
                selected_features = [col for col, var in self.ml_feature_vars.items() if var.get()]
            
            target = self.ml_target_combo.get() if hasattr(self, 'ml_target_combo') and self.ml_target_combo.get() else ""
            
            # Create preview dataframe with selected columns
            preview_cols = selected_features.copy()
            if target and target in df_current.columns:
                preview_cols.append(target)
                
            if preview_cols:
                try:
                    preview_df = df_current[preview_cols].head(20)
                    self.ml_preview_frame.set_dataframe(preview_df)
                except Exception as e:
                    self.ml_preview_frame.set_text(f"Error displaying preview: {str(e)}")
            else:
                self.ml_preview_frame.set_text("No features selected")

    def _clear_ml_logs(self):
        """Clear ML logs"""
        if hasattr(self, 'ml_logs_text'):
            self.ml_logs_text.delete(1.0, tk.END)

    def _log_ml_message(self, message):
        """Add message to ML logs"""
        if hasattr(self, 'ml_logs_text'):
            self.ml_logs_text.insert(tk.END, f"{message}\n")
            self.ml_logs_text.see(tk.END)
            self.ml_logs_text.update()

    def _run_pycaret_training(self):
        """Run PyCaret training WITHOUT threading"""
        # Disable button to prevent multiple clicks
        if hasattr(self, 'apply_model_btn'):
            self.apply_model_btn.config(state="disabled")
            self.apply_model_btn.config(text="Training...")
            self.root.update_idletasks()
        
        # Use after() to run training without blocking
        self.root.after(100, self._run_pycaret_sync)

    def _run_pycaret_sync(self):
        """Run PyCaret training synchronously without threading"""
        try:
            # Collect parameters
            selected_features = [col for col, var in getattr(self, 'ml_feature_vars', {}).items() if var.get()]
            selected_features = list(dict.fromkeys(selected_features))  # Remove duplicates
            
            # Get target
            raw_target = self.ml_target_combo.get() if hasattr(self, 'ml_target_combo') else ""
            
            # FIX: Convert "Space" or empty string to None (For Clustering)
            if raw_target.strip() == "": 
                target = None  # Unsupervised / Clustering
            else:
                target = raw_target
            
            # Get current dataframe
            df_processed = getattr(self, "df_processed", None)
            df_original = getattr(self, "df_original", None)
            
            if df_processed is not None:
                df_current = df_processed
            elif df_original is not None:
                df_current = df_original
            else:
                messagebox.showerror("No Data", "Please load data first.")
                return
            
            # Validation checks
            if not selected_features:
                messagebox.showerror("No Features", "Please select at least one feature.")
                return
            
            # Fix duplicate column names
            df_current = df_current.copy()
            if not df_current.columns.is_unique:
                new_columns = []
                counter = {}
                for col in df_current.columns:
                    if col in counter:
                        counter[col] += 1
                        new_name = f"{col}_{counter[col]}"
                        new_columns.append(new_name)
                    else:
                        counter[col] = 0
                        new_columns.append(col)
                df_current.columns = new_columns
            
            # Import PyCaret trainer
            try:
                from trainer_pycaret import PyCaretTrainer
            except ImportError as e:
                messagebox.showerror("Import Error", f"PyCaret trainer not available: {str(e)}")
                return
            
            output_dir = getattr(self, 'OUTPUT_DIR', OUTPUT_DIR)
            trainer = PyCaretTrainer(df_current, selected_features, target, output_dir)
            
            # Logging setup
            def log_to_ui(message):
                self._log_ml_message(message)
                self.root.update_idletasks()
            trainer.log_queue = type('SimpleQueue', (), {'put': log_to_ui})()
            
            # --- VALIDATION SETTINGS LOGIC ---
            test_size = self.test_size_var.get()
            use_manual_split = self.show_validation_var.get()
            
            if not use_manual_split:
                self._log_ml_message("ℹ️ Validation hidden: Using ALL data for training (auto-split in PyCaret).")
                # Setting test_size to 0 in params effectively tells trainer logic to skip manual splitting
                test_size = 0.0 
            
            selected_model = self.model_selection.get().strip()
            use_compare = True
            
            # CRITICAL FIX: If user selected 'xgboost' or 'lightgbm', TURN OFF COMPARISON
            if selected_model and selected_model != " ":
                use_compare = False
                self._log_ml_message(f"🔒 Locked to Model: {selected_model.upper()}")
            else:
                self._log_ml_message("🔍 No specific model selected. Running Comparison...")

            # Collect training parameters
# Collect training parameters
            training_params = {
                'problem_kind': self.problem_type.get(),
                'use_compare': use_compare,
                'model_to_create': selected_model,
                'tune': self.tune_model.get(),
                'finalize': self.finalize_model.get(),
                'use_nan_as_test': getattr(self, 'use_nan_test', tk.BooleanVar(value=False)).get(),
                'fold': getattr(self, 'cv_folds', tk.IntVar(value=5)).get(),
                'session_id': getattr(self, 'session_id', tk.IntVar(value=42)).get(),
                'test_size': test_size,
                'random_state': getattr(self, 'random_state_var', tk.IntVar(value=42)).get(),
                
                # --- UPDATED SETUP KWARGS ---
                'setup_kwargs': {
                    'html': False,
                    'verbose': False,
                    'system_log': False,
                    
                    # POLYNOMIAL CONFIGURATION
                    'polynomial_features': self.poly_features.get(),
                    'polynomial_degree': self.poly_degree.get(),
                    
                    # Safety: Remove highly correlated features created by polynomials
                    'remove_multicollinearity': True, 
                    'multicollinearity_threshold': 0.95
                }
                # ----------------------------
            }
            
            # Add hyperparameters if available
            if selected_model and selected_model != " " and hasattr(self, 'model_hyperparameters'):
                if selected_model in self.model_hyperparameters:
                    training_params['model_hyperparameters'] = self.model_hyperparameters[selected_model]
            
            self._log_ml_message("⏳ Starting model training...")
            self.root.update_idletasks()
            
            # RUN TRAINING
            results = trainer.run(**training_params)
            
            if "error" in results:
                self._handle_pycaret_error(results["error"])
            else:
                self._handle_pycaret_results(results)
                
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self._log_ml_message(f"❌ {error_msg}")
            messagebox.showerror("Training Error", error_msg)
        finally:
            if hasattr(self, 'apply_model_btn'):
                self.apply_model_btn.config(state="normal")
                self.apply_model_btn.config(text="Apply Model")

    def _run_pycaret_training_thread_safe(self, thread_params):
        """Thread-safe version of training thread"""
        try:
            from trainer_pycaret import PyCaretTrainer
            
            output_dir = getattr(self, 'OUTPUT_DIR', OUTPUT_DIR)
            
            # Create trainer with thread-safe logging
            trainer = PyCaretTrainer(
                thread_params['df_current'], 
                thread_params['selected_features'], 
                thread_params['target'], 
                output_dir
            )
            
            # Thread-safe logging using our ThreadSafeTkinter class
            def log_to_ui(message):
                self.thread_safe.safe_call(self._log_ml_message, message)
            
            # Pass the thread-safe logger to the trainer
            trainer.log_queue = self.thread_safe
            
            # Run training
            results = trainer.run(
                problem_kind=thread_params['problem_type'],
                use_compare=(thread_params['experiment_type'] == "compare"),
                model_to_create=thread_params['model_selection'],
                tune=thread_params['tune_model'],
                finalize=thread_params['finalize_model'],
                use_nan_as_test=thread_params.get('use_nan_test', False),
                fold=thread_params['cv_folds'],
                session_id=thread_params['session_id'],
                test_size=thread_params['test_size'],
                random_state=thread_params['random_state']
            )
            
            # Thread-safe result handling
            if 'error' in results:
                self.thread_safe.safe_call(self._handle_pycaret_error, results['error'])
            else:
                self.thread_safe.safe_call(self._handle_pycaret_results, results)
                
        except Exception as e:
            error_msg = f"Training thread error: {str(e)}"
            print(f"THREAD ERROR: {error_msg}")
            self.thread_safe.safe_call(self._handle_pycaret_error, error_msg)

    def _redirect_pycaret_logs(self):
        """Redirect PyCaret logs to our GUI"""
        import sys
        from io import StringIO
        
        class PyCaretOutputRedirector:
            def __init__(self, log_callback):
                self.log_callback = log_callback
                self.buffer = StringIO()
                
            def write(self, text):
                if text.strip():
                    self.log_callback(text.strip())
                self.buffer.write(text)
                
            def flush(self):
                pass
        
        # Create redirector
        self.pycaret_redirector = PyCaretOutputRedirector(self._log_ml_message)
        
        # Redirect stdout for PyCaret
        sys.stdout = self.pycaret_redirector

    def _handle_pycaret_results(self, results):
        """Handle PyCaret training results with better logging"""
        # Re-enable button
        self.apply_model_btn.config(state="normal")
        self.apply_model_btn.config(text="Apply Model")
        
        if "error" in results:
            self._log_ml_message(f"❌ Training failed: {results['error']}")
            messagebox.showerror("Training Error", results["error"])
        else:
            self._log_ml_message("✅ Training completed successfully!")
            self.last_pycaret_results = results
            
            # Store test data for visualization
            self.test_data = (
                results.get('test_data'), 
                results.get('y_true')
            )
            self.train_data = results.get('train_data')
            self.y_true = results.get('y_true')
            self.y_pred = results.get('y_pred')
            
            # Show detailed results
            self._update_results_display(results)
            
            # Automatically generate and display plots
            self._refresh_results_plots()
            
            # Show success message
            model_type = type(results.get('model')).__name__ if results.get('model') else "Unknown"
            self._log_ml_message(f"📦 Model: {model_type}")
            
            if results.get("artifacts", {}).get("model_path"):
                self._log_ml_message(f"💾 Model saved to: {results['artifacts']['model_path']}")

        # Refresh feature dropdown in results tab
        if hasattr(self, 'feature_selection') and hasattr(self, 'last_pycaret_results'):
            features = self.last_pycaret_results.get('features', [])
            # Find the feature combo and update its values
            for widget in self.tab_results.winfo_children():
                if isinstance(widget, ttk.Panedwindow):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for subchild in child.winfo_children():
                                if isinstance(subchild, ttk.LabelFrame) and subchild.cget('text') == 'Visualization Controls':
                                    for control in subchild.winfo_children():
                                        if isinstance(control, ttk.Frame):
                                            for item in control.winfo_children():
                                                if isinstance(item, ttk.Combobox):
                                                    item.config(values=features)
                                                    if features:
                                                        item.set(features[0])
        self._update_feature_combo()
        
    def _update_results_display(self, results):
            """Update results display using the nice table"""
            # Use the new table formatter
            self._display_metrics_table(results)
            
            # You can add extra details below if needed, like the artifact paths
            artifacts = results.get('artifacts', {})
            if artifacts:
                self.results_text.insert("end", "\n💾 SAVED FILES:\n")
                for k, v in artifacts.items():
                    self.results_text.insert("end", f"  • {k}: {os.path.basename(v)}\n")
                
    def _handle_pycaret_error(self, error_msg):
        """Handle PyCaret training errors"""
        # FIX: Use apply_model_btn instead of run_ml_btn
        if hasattr(self, 'apply_model_btn'):
            self.apply_model_btn.config(state="normal")
            self.apply_model_btn.config(text="Apply Model")
        
        self._log_ml_message(f"Training error: {error_msg}")
        messagebox.showerror("Training Error", error_msg)

    def _generate_plots(self):
        """Generate selected visualizations for the trained model"""
        if not hasattr(self, 'last_pycaret_results') or not self.last_pycaret_results.get('model'):
            messagebox.showerror("No Model", "Please train a model first.")
            return
        
        try:
            model = self.last_pycaret_results['model']
            problem_type = self.problem_type.get()
            
            # Import appropriate plotting functions based on problem type
            if problem_type == "classification":
                from pycaret.classification import plot_model
            elif problem_type == "regression":
                from pycaret.regression import plot_model
            else:
                messagebox.showinfo("Not Supported", "Visualizations are only available for classification and regression.")
                return
            
            # Generate selected plots
            for plot_name, var in self.viz_options.items():
                if var.get():
                    try:
                        plot_model(model, plot=plot_name, save=True, scale=2)
                        self._log_ml_message(f"Generated {plot_name} plot")
                    except Exception as e:
                        self._log_ml_message(f"Failed to generate {plot_name}: {str(e)}")
            
            messagebox.showinfo("Plots Generated", "Selected plots have been saved to the output folder.")
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plots: {str(e)}")

    def _export_pycaret_model(self):
        """Export the trained PyCaret model"""
        if not hasattr(self, 'last_pycaret_results'):
            messagebox.showinfo("No Model", "No trained model available.")
            return
        
        model_path = self.last_pycaret_results.get("artifacts", {}).get("model_path")
        if not model_path:
            messagebox.showinfo("No Model", "No model path found in results.")
            return
        
        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Save PyCaret Model"
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy2(model_path, save_path)
                messagebox.showinfo("Success", f"Model exported to:\n{save_path}")
                self._log_ml_message(f"Model exported to: {save_path}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export model: {str(e)}")

    def _open_advanced_ml(self):
        """Open the advanced ML dialog"""
        if self.df_processed is None:
            messagebox.showerror("No Data", "Load and preprocess data first.")
            return
        
        # Get selected features and target
        features = [c for c, v in getattr(self, "feature_vars", {}).items() if v.get()]
        target = self.target_combo.get() if hasattr(self, "target_combo") else None
        
        if not features:
            messagebox.showerror("No Features", "Select features in Preprocess tab first.")
            return
        
        # Create advanced trainer
        from trainer_advanced import AdvancedModelTrainer
        from ml_advanced_ui import AdvancedMLUI
        
        trainer = AdvancedModelTrainer(self.df_processed, features, target)
        
        # Open advanced UI
        AdvancedMLUI(self.root, trainer, self._on_advanced_model_trained)

    def _on_advanced_model_trained(self, results):
        """Handle results from advanced ML training"""
        if "error" in results:
            messagebox.showerror("Training Error", results["error"])
            return
        
        # Display results
        self.last_results = results
        if hasattr(self, 'log_text'):
            self.log_text.insert("end", f"\n=== Advanced ML Results ===\n")
            
            metrics = results.get("metrics", {})
            for metric, value in metrics.items():
                self.log_text.insert("end", f"{metric}: {value:.4f}\n")
            
            self.log_text.insert("end", f"\nAdvanced training completed!\n")
        self.status.config(text="Advanced ML training completed!")

    # ----------------------------
    # RESULTS TAB (vertical paned)
    # ----------------------------

    def _build_results_tab(self):
            """Build the results tab with interpretation tools"""
            for w in self.tab_results.winfo_children(): w.destroy()
            
            paned = ttk.Panedwindow(self.tab_results, orient="horizontal")
            paned.pack(fill="both", expand=True)
            
            # ==================== LEFT: Controls & Metrics ====================
            left_frame = ttk.Frame(paned, width=350)
            left_frame.pack_propagate(False)
            
            # Metrics summary
            metrics_frame = ttk.LabelFrame(left_frame, text="Model Metrics", padding=10)
            metrics_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            self.metrics_text = tk.Text(metrics_frame, height=10, wrap="word")
            metrics_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metrics_text.yview)
            self.metrics_text.configure(yscrollcommand=metrics_scrollbar.set)
            self.metrics_text.pack(side="left", fill="both", expand=True)
            metrics_scrollbar.pack(side="right", fill="y")
            
            # --- NEW: MODEL INTERPRETATION ---
            interp_frame = ttk.LabelFrame(left_frame, text="Model Interpretation (Nature)", padding=10)
            interp_frame.pack(fill="x", padx=10, pady=10)
            
            ttk.Button(interp_frame, text="📜 Show Equation / Rules (Text)", 
                    command=self._display_model_equation_or_rules).pack(fill="x", pady=2)
                    
            ttk.Button(interp_frame, text="🌳 Show Decision Tree (Plot)", 
                    command=lambda: self._generate_nature_plot("tree")).pack(fill="x", pady=2)

            ttk.Button(interp_frame, text="📊 Show Coefficients / Importance", 
                    command=lambda: self._generate_nature_plot("importance")).pack(fill="x", pady=2)

            # General Visualizations
            viz_controls = ttk.LabelFrame(left_frame, text="General Visualizations", padding=10)
            viz_controls.pack(fill="x", padx=10, pady=10)
            
            ttk.Button(viz_controls, text="Refresh All Plots", command=self._refresh_results_plots).pack(fill="x", pady=2)
                       
            # Export
            export_frame = ttk.LabelFrame(left_frame, text="Export", padding=10)
            export_frame.pack(fill="x", padx=10, pady=10)
            ttk.Button(export_frame, text="Save Session", command=self._save_analysis_session).pack(fill="x", pady=2)
            
            paned.add(left_frame, weight=1)

            # ==================== RIGHT: Plot & Text Display ====================
            right_paned = ttk.Panedwindow(paned, orient="vertical")
            
            # Plot display area
            plot_frame = ttk.LabelFrame(right_paned, text="Visualizations", padding=10)
            
            nav_frame = ttk.Frame(plot_frame)
            nav_frame.pack(fill="x", pady=5)
            ttk.Button(nav_frame, text="← Previous", command=self._previous_plot).pack(side="left", padx=5)
            ttk.Button(nav_frame, text="Next →", command=self._next_plot).pack(side="left", padx=5)
            self.plot_nav_label = ttk.Label(nav_frame, text="No plots")
            self.plot_nav_label.pack(side="left", padx=10)
            ttk.Button(nav_frame, text="Interactive View", command=self._open_interactive_view).pack(side="right", padx=5)
            
            self.plot_canvas_frame = ttk.Frame(plot_frame)
            self.plot_canvas_frame.pack(fill="both", expand=True, pady=5)
            
            # Rules/Equation Text Display Area (Bottom Right)
            self.rules_display_frame = ttk.LabelFrame(right_paned, text="Model Logic / Description", padding=10)
            
            self.rules_text = tk.Text(self.rules_display_frame, height=10, wrap="word", font=("Consolas", 10))
            rules_scroll = ttk.Scrollbar(self.rules_display_frame, orient="vertical", command=self.rules_text.yview)
            self.rules_text.configure(yscrollcommand=rules_scroll.set)
            self.rules_text.pack(side="left", fill="both", expand=True)
            rules_scroll.pack(side="right", fill="y")

            right_paned.add(plot_frame, weight=3)
            right_paned.add(self.rules_display_frame, weight=1)
            paned.add(right_paned, weight=3)
            
            self.current_plot_index = 0
            self.generated_plots = {}

    def _update_feature_combo(self):
        """Update the feature combo box with available features"""
        if hasattr(self, 'last_pycaret_results') and hasattr(self, 'feature_combo'):
            features = self.last_pycaret_results.get('features', [])
            self.feature_combo.config(values=features)
            if features:
                self.feature_combo.set(features[0])

    def _refresh_results_plots(self):
        """Refresh results with Force-Injected Target for Clustering"""
        if not hasattr(self, 'last_pycaret_results'): return
        
        try:
            for widget in self.plot_canvas_frame.winfo_children(): widget.destroy()
            output_dir = os.path.join(getattr(self, 'OUTPUT_DIR', 'output'), "results_plots")
            os.makedirs(output_dir, exist_ok=True)
            
            model_pipeline = self.last_pycaret_results.get('model')
            feature_names = self.last_pycaret_results.get('features', [])
            problem_type = self.last_pycaret_results.get('problem_type', 'regression')
            
            train_data = self.last_pycaret_results.get('train_data')
            target_col = self.last_pycaret_results.get('target')

            # --- FORCE INJECT TARGET FOR CLUSTERING ---
            # If we are clustering, we want the visualizer to see the Target (Price) too!
            if problem_type == 'clustering' and target_col and train_data is not None:
                if isinstance(train_data, pd.DataFrame) and target_col not in train_data.columns:
                    # Try to find the target data in our processed dataframe
                    if hasattr(self, 'df_processed') and target_col in self.df_processed.columns:
                        # Add it back temporarily for visualization
                        try:
                            # We need to align indices. This is a bit hacky but necessary if Trainer dropped it.
                            # Safest way: merge by index
                            target_series = self.df_processed[target_col]
                            train_data = train_data.merge(target_series, left_index=True, right_index=True, how='left')
                            if target_col not in feature_names:
                                feature_names.append(target_col)
                            self._log_ml_message(f"➕ Injected '{target_col}' into Clustering Plot data")
                        except: pass
            # -------------------------------------------

            from enhanced_visualizer import EnhancedVisualizer
            visualizer = EnhancedVisualizer(
                self.last_pycaret_results,
                problem_type,
                model_pipeline,
                self.test_data,
                train_data, # Now contains Target
                feature_names,
                target_col
            )
            
            self._log_ml_message("📊 Generating plots...")
            self.generated_plots = visualizer.create_comprehensive_plots(output_dir)
            
            # Clean up
            self.generated_plots = {k: v for k, v in self.generated_plots.items() if v is not None}
            
            if self.generated_plots:
                self.current_plot_index = 0
                self._display_current_plot()
                self._update_results_list()
                self.status.config(text=f"Generated {len(self.generated_plots)} plots")
            else:
                self.status.config(text="No plots generated")
                
        except Exception as e:
            self._log_ml_message(f"❌ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_dynamic_comparison(self, feature_name):
            """Create dynamic train/test comparison for a specific feature"""
            if not hasattr(self, 'enhanced_visualizer'):
                # Re-initialize visualizer if missing, ensuring we pass the correct feature list
                self._refresh_results_plots()
                
            if not hasattr(self, 'enhanced_visualizer'):
                messagebox.showinfo("No Visualizer", "Please generate plots first.")
                return
            
            # CRITICAL FIX: Ensure we are using the TRANSFORMED feature list from results
            # This ensures "feature_0" is mapped back to "Manufacturer_Ford"
            if hasattr(self, 'last_pycaret_results'):
                transformed_features = self.last_pycaret_results.get('features', [])
                if transformed_features:
                    # Update the visualizer's knowledge of features
                    self.enhanced_visualizer.features = transformed_features

            output_dir = os.path.join(getattr(self, 'OUTPUT_DIR', OUTPUT_DIR), "results_plots")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate the plot
            plot_path = self.enhanced_visualizer.create_dynamic_comparison(output_dir, feature_name)
            
            if plot_path:
                # Add to generated plots and display
                plot_name = f"Feature Analysis: {feature_name}"
                self.generated_plots[plot_name] = plot_path
                self.current_plot_index = len(self.generated_plots) - 1
                self._display_current_plot()
                self._log_ml_message(f"✅ Generated: {plot_name}")
            else:
                messagebox.showinfo("Error", f"Could not generate comparison for {feature_name}")

    def _update_metrics_display(self):
        """Update the metrics display"""
        if not hasattr(self, 'last_pycaret_results'):
            return
            
        self.metrics_text.delete(1.0, tk.END)
        results = self.last_pycaret_results
        
        # Model info
        model = results.get('model')
        if model:
            self.metrics_text.insert(tk.END, f"Model Type: {type(model).__name__}\n\n")
        
        # Metrics
        metrics = results.get('metrics', {})
        if metrics:
            self.metrics_text.insert(tk.END, "=== PERFORMANCE METRICS ===\n\n")
            for metric, value in metrics.items():
                if metric not in ['run_log', 'test_predictions_made']:
                    if isinstance(value, float):
                        self.metrics_text.insert(tk.END, f"{metric}: {value:.4f}\n")
                    else:
                        self.metrics_text.insert(tk.END, f"{metric}: {value}\n")
        
        # Training info
        if 'run_log' in results:
            logs = results['run_log']
            # Extract key information from logs
            lines = logs.split('\n')
            key_info = [line for line in lines if any(keyword in line.lower() for keyword in 
                        ['final data shapes', 'best model', 'test set metrics', 'model saved'])]
            
            if key_info:
                self.metrics_text.insert(tk.END, "\n=== TRAINING SUMMARY ===\n\n")
                for info in key_info:
                    self.metrics_text.insert(tk.END, f"{info}\n")



    def _customize_plot(self):
        """Open plot customization dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Customize Plot")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Plot Customization", font=("", 12, "bold")).pack(pady=10)
        
        # Title customization
        title_frame = ttk.Frame(dialog)
        title_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(title_frame, text="Title:").pack(side="left")
        title_var = tk.StringVar(value="Model Results")
        ttk.Entry(title_frame, textvariable=title_var).pack(side="left", fill="x", expand=True, padx=5)
        
        # Axis labels
        xaxis_frame = ttk.Frame(dialog)
        xaxis_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(xaxis_frame, text="X-Axis:").pack(side="left")
        xaxis_var = tk.StringVar(value="X")
        ttk.Entry(xaxis_frame, textvariable=xaxis_var).pack(side="left", fill="x", expand=True, padx=5)
        
        yaxis_frame = ttk.Frame(dialog)
        yaxis_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(yaxis_frame, text="Y-Axis:").pack(side="left")
        yaxis_var = tk.StringVar(value="Y")
        ttk.Entry(yaxis_frame, textvariable=yaxis_var).pack(side="left", fill="x", expand=True, padx=5)
        
        # Apply button
        def apply_customization():
            # Here you would regenerate the plot with custom titles
            messagebox.showinfo("Customization", "Plot customization would be applied here")
            dialog.destroy()
        
        ttk.Button(dialog, text="Apply Customization", command=apply_customization).pack(pady=20)

    def _save_all_plots(self):
        """Save all generated plots"""
        if not self.generated_plots:
            messagebox.showinfo("No Plots", "No plots to save.")
            return
        
        # Create plots subdirectory
        plots_dir = os.path.join(getattr(self, 'OUTPUT_DIR', OUTPUT_DIR), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            for plot_name, plot_path in self.generated_plots.items():
                # Copy plot to plots directory
                import shutil
                dest_path = os.path.join(plots_dir, os.path.basename(plot_path))
                shutil.copy2(plot_path, dest_path)
            
            messagebox.showinfo("Success", f"All plots saved to:\n{plots_dir}")
            self._open_folder(plots_dir)
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plots: {str(e)}")

    def _update_results_list(self):
        """Update the results files list"""
        if hasattr(self, 'results_files_list'):
            self.results_files_list.delete(0, tk.END)
            
            output_dir = getattr(self, 'OUTPUT_DIR', OUTPUT_DIR)
            if not os.path.exists(output_dir):
                return
                
            # Add all result files
            for file in sorted(os.listdir(output_dir)):
                if file.endswith(('.png', '.jpg', '.jpeg', '.pkl', '.csv')):
                    self.results_files_list.insert(tk.END, file)

    def _open_selected_result_file(self):
        """Open the selected result file"""
        selection = self.results_files_list.curselection()
        if not selection:
            return
            
        filename = self.results_files_list.get(selection[0])
        output_dir = getattr(self, 'OUTPUT_DIR', OUTPUT_DIR)
        filepath = os.path.join(output_dir, filename)
        
        try:
            if os.name == "nt":
                os.startfile(filepath)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", filepath])
        except Exception as e:
            messagebox.showerror("Open Failed", f"Could not open file: {str(e)}")

    def _open_selected_output(self):
        sel = self.outputs_list.curselection()
        if not sel:
            return
        fname = self.outputs_list.get(sel[0])
        path = os.path.join(OUTPUT_DIR, fname)
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _export_metrics_csv(self):
        if not hasattr(self, "last_results"):
            messagebox.showinfo("No metrics", "No training performed yet.")
            return
        metrics = self.last_results.get("metrics", {})
        import csv
        out = os.path.join(OUTPUT_DIR, f"metrics_{safe_filename(str(pd.Timestamp.now()))}.csv")
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        messagebox.showinfo("Exported", f"Metrics exported to {out}")
        self._update_results_list()

    # ----------------------------
    # Helpers
        # ----------------------------
    def _open_output_dir(self):
        """Open the output directory in file explorer"""
        try:
            output_dir = getattr(self, 'OUTPUT_DIR', OUTPUT_DIR)
            if not os.path.exists(output_dir):
                ensure_dir(output_dir)
            
            if os.name == "nt":
                os.startfile(output_dir)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", output_dir])
        except Exception as e:
            messagebox.showinfo("Output Folder", f"Output directory:\n{output_dir}\n\nError opening: {str(e)}")

    def _show_help(self):
            """Open a comprehensive Help & Troubleshooting window"""
            help_win = tk.Toplevel(self.root)
            help_win.title("Help & Troubleshooting Guide")
            help_win.geometry("800x600")
            
            # Create scrollable text area
            from tkinter import scrolledtext
            text_area = scrolledtext.ScrolledText(help_win, wrap="word", width=90, height=30, font=("Consolas", 10))
            text_area.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Define tags for formatting
            text_area.tag_config("header", font=("Consolas", 12, "bold"), foreground="#00008B")
            text_area.tag_config("error", font=("Consolas", 10, "bold"), foreground="#8B0000")
            text_area.tag_config("fix", font=("Consolas", 10, "bold"), foreground="#006400")
            
            # --- CONTENT GENERATION ---
            
            # 1. Quick Guide
            text_area.insert("end", "=== 🚀 QUICK START GUIDE ===\n\n", "header")
            text_area.insert("end", 
                "1. DATA TAB:\n"
                "   - Load CSV/Excel files via Local Path, URL, or Google Drive.\n"
                "   - Use the Directory Viewer to double-click files for quick loading.\n\n"
                "2. PREPROCESS TAB:\n"
                "   - Select your 'Target' (what you want to predict).\n"
                "   - Click '...' next to columns to clean specific data (Map values, Remove outliers).\n"
                "   - Use 'Advanced Tools' for PCA or Custom Python Scripts.\n"
                "   - ALWAYS click 'Apply Preprocessing' before moving to ML.\n\n"
                "3. ML APPLICATION TAB:\n"
                "   - Select Task (Classification/Regression) and Model.\n"
                "   - Use 'Configure Hyperparameters' for fine-tuning.\n"
                "   - Toggle 'Show Advanced Validation' to set Train/Test split size.\n\n"
                "4. RESULTS TAB:\n"
                "   - View Metrics (Train vs Test) to check for overfitting.\n"
                "   - Generate Plots (Confusion Matrix, Decision Tree, etc.).\n"
                "   - Export your model (.pkl) and results.\n\n"
            )
            
            # 2. Troubleshooting (The Errors we fixed)
            text_area.insert("end", "=== 🔧 TROUBLESHOOTING COMMON ERRORS ===\n\n", "header")
            
            # Error 1: Single Class
            text_area.insert("end", "❌ Error: 'The least populated class in y has only 1 member'\n", "error")
            text_area.insert("end", "   Cause: Your Target column has a value that appears only once (e.g., one specific car brand).\n")
            text_area.insert("end", "   ✅ Fix 1: Go to Preprocess > Advanced Tools > Python Script.\n")
            text_area.insert("end", "   ✅ Fix 2: Run a script to remove rare values (df = df[df['Target'].map(df['Target'].value_counts()) > 1]).\n")
            text_area.insert("end", "   ✅ Fix 3: Or use 'Row Filters > Auto-Remove Rare Values' in the Column Editor.\n\n")

            # Error 2: Shutil / Save Error
            text_area.insert("end", "❌ Error: 'local variable shutil referenced before assignment'\n", "error")
            text_area.insert("end", "   Cause: An import issue in the save function.\n")
            text_area.insert("end", "   ✅ Fix: This has been patched in the latest version. Try saving again.\n\n")

            # Error 3: PCA Text Error
            text_area.insert("end", "❌ Error: 'could not convert string to float' in PCA\n", "error")
            text_area.insert("end", "   Cause: You tried to run PCA on text columns (like 'Mileage' with 'km' inside).\n")
            text_area.insert("end", "   ✅ Fix: In Preprocess tab, use 'Strip Substring' to remove 'km', then convert type to Float/Int.\n\n")

            # Error 4: Pipeline/Tree Rules
            text_area.insert("end", "❌ Error: 'Pipeline object has no attribute tree_'\n", "error")
            text_area.insert("end", "   Cause: PyCaret wraps models in a pipeline. The visualizer couldn't find the tree.\n")
            text_area.insert("end", "   ✅ Fix: Code updated to automatically extract 'actual_estimator' from pipeline.\n\n")

            # Error 5: Missing Target in PCA
            text_area.insert("end", "❌ Issue: Target column disappears after PCA\n", "error")
            text_area.insert("end", "   Cause: PCA was applied to the Target column, transforming it into numbers.\n")
            text_area.insert("end", "   ✅ Fix: Select your Target in the Preprocess tab BEFORE opening Advanced Tools. The app now protects the target.\n\n")

            # Error 6: UI Freezing
            text_area.insert("end", "❌ Issue: App freezes during training\n", "error")
            text_area.insert("end", "   Cause: Machine Learning is heavy math running on your processor.\n")
            text_area.insert("end", "   ✅ Fix: Check the terminal/console for background progress. Be patient with large datasets.\n\n")

            # 3. Tips
            text_area.insert("end", "=== 💡 PRO TIPS ===\n\n", "header")
            text_area.insert("end", "• Overfitting? If Train Score is 99% but Test Score is 60%, your model memorized the data. Try reducing 'Max Depth' in Hyperparameters.\n")
            text_area.insert("end", "• Small Data? If you have <50 rows, uncheck 'Show Validation' to train on 100% of data.\n")
            text_area.insert("end", "• Polynomials? Enable 'Polynomial Features' in ML tab to fit curves instead of straight lines.\n")

            text_area.config(state="disabled") # Make read-only
            
            # Close button
            ttk.Button(help_win, text="Close Help", command=help_win.destroy).pack(pady=5)

    def _debug_settings(self, per_col, global_settings):
        """Debug method to see what settings are being used"""
        print("=== DEBUG SETTINGS ===")
        print("Global Settings:")
        for key, value in global_settings.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        # Specifically check missing value removal
        print(f"  remove_all_missing enabled: {global_settings.get('remove_all_missing', False)}")
        
        print("Per-Column Settings:")
        for col, settings in per_col.items():
            has_settings = any(settings.values())
            if has_settings:
                print(f"  {col}:")
                for key, value in settings.items():
                    if value:
                        print(f"    {key}: {value}")
        print("=====================")
        
    def _show_transformation_summary(self):
        """Show a summary of applied transformations"""
        if not hasattr(self, 'preprocessor'):
            return
            
        logs = self.preprocessor.get_transformations_log()
        if logs:
            summary = "Recent Transformations:\n"
            for log in logs[-5:]:
                summary += f"• {log}\n"
            
            # Show in status bar or as a tooltip
            self.status.config(text=summary if len(summary) < 100 else summary[:100] + "...")
            
            # Also print to console for debugging
            print("=== TRANSFORMATION LOGS ===")
            for log in logs:
                print(f"• {log}")
            print("===========================")

    # Add this method to your App class in main.py

    def _display_current_plot(self):
        """Display the current plot in the canvas"""
        if not self.generated_plots or self.current_plot_index >= len(self.generated_plots):
            # Show message when no plots
            for widget in self.plot_canvas_frame.winfo_children():
                widget.destroy()
            no_plot_label = ttk.Label(self.plot_canvas_frame, text="No plots to display")
            no_plot_label.pack(expand=True)
            
            if hasattr(self, 'plot_nav_label'):
                self.plot_nav_label.config(text="No plots generated")
            return
            
        # Clear existing canvas
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()
        
        try:
            plot_path = list(self.generated_plots.values())[self.current_plot_index]
            plot_name = list(self.generated_plots.keys())[self.current_plot_index]
            
            # Display image
            from PIL import Image, ImageTk
            img = Image.open(plot_path)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = ttk.Label(self.plot_canvas_frame, image=photo)
            label.image = photo  # Keep a reference
            label.pack(expand=True)
            
            # Update navigation - THIS IS CRITICAL
            if hasattr(self, 'plot_nav_label'):
                self.plot_nav_label.config(
                    text=f"Plot {self.current_plot_index + 1} of {len(self.generated_plots)}: {plot_name}"
                )
            
        except Exception as e:
            error_label = ttk.Label(self.plot_canvas_frame, text=f"Error displaying plot: {str(e)}")
            error_label.pack(expand=True)

    def _previous_plot(self):
        """Show previous plot"""
        if self.generated_plots and self.current_plot_index > 0:
            self.current_plot_index -= 1
            self._display_current_plot()

    def _next_plot(self):
        """Show next plot"""
        if self.generated_plots and self.current_plot_index < len(self.generated_plots) - 1:
            self.current_plot_index += 1
            self._display_current_plot()

    def _generate_adv_selected_graphs(self, visualizer):
        """Generate selected graphs in advanced interface"""
        selected_graphs = [graph_type for graph_type, var in self.adv_graph_vars.items() if var.get()]
        
        if not selected_graphs:
            messagebox.showinfo("No Selection", "Please select at least one graph type.")
            return
        
        output_dir = os.path.join(self.OUTPUT_DIR, "advanced_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        generated_plots = {}
        
        for graph_type in selected_graphs:
            try:
                plot_path = visualizer.generate_graph(graph_type, output_dir)
                if plot_path:
                    graph_name = visualizer.get_available_graphs().get(graph_type, graph_type)
                    generated_plots[graph_name] = plot_path
            except Exception as e:
                print(f"Failed to generate {graph_type}: {e}")
        
        if generated_plots:
            self.adv_generated_plots = generated_plots
            self.adv_current_plot_index = 0
            self._adv_display_current_plot()
            self._refresh_gallery()
            messagebox.showinfo("Success", f"Generated {len(generated_plots)} plots!")
        else:
            messagebox.showinfo("Error", "No plots were generated.")

    def _generate_adv_all_graphs(self, visualizer):
        """Generate all possible graphs"""
        output_dir = os.path.join(self.OUTPUT_DIR, "advanced_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        self.adv_generated_plots = visualizer.generate_all_possible_graphs(output_dir)
        
        if self.adv_generated_plots:
            self.adv_current_plot_index = 0
            self._adv_display_current_plot()
            self._refresh_gallery()
            messagebox.showinfo("Success", f"Generated {len(self.adv_generated_plots)} plots!")
        else:
            messagebox.showinfo("Error", "No plots were generated.")

    def _generate_adv_single_graph(self, visualizer, graph_type):
        """Generate a single graph"""
        output_dir = os.path.join(self.OUTPUT_DIR, "advanced_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = visualizer.generate_graph(graph_type, output_dir)
        
        if plot_path:
            graph_name = visualizer.get_available_graphs().get(graph_type, graph_type)
            self.adv_generated_plots[graph_name] = plot_path
            self.adv_current_plot_index = len(self.adv_generated_plots) - 1
            self._adv_display_current_plot()
            self._refresh_gallery()
            messagebox.showinfo("Success", f"Generated {graph_name}!")
        else:
            messagebox.showinfo("Error", f"Could not generate {graph_type}")

    def _generate_adv_combo(self, visualizer, graph_types, combo_name):
        """Generate a combination of graphs"""
        output_dir = os.path.join(self.OUTPUT_DIR, "advanced_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        generated_plots = {}
        
        for graph_type in graph_types:
            try:
                plot_path = visualizer.generate_graph(graph_type, output_dir)
                if plot_path:
                    graph_name = visualizer.get_available_graphs().get(graph_type, graph_type)
                    generated_plots[graph_name] = plot_path
            except:
                continue
        
        if generated_plots:
            self.adv_generated_plots.update(generated_plots)
            self.adv_current_plot_index = len(self.adv_generated_plots) - 1
            self._adv_display_current_plot()
            self._refresh_gallery()
            messagebox.showinfo("Success", f"Generated {combo_name} with {len(generated_plots)} plots!")
        else:
            messagebox.showinfo("Error", f"Could not generate {combo_name}")

    def _adv_display_current_plot(self):
        """Display current plot in advanced interface"""
        if not self.adv_generated_plots:
            for widget in self.adv_plot_frame.winfo_children():
                widget.destroy()
            self.adv_nav_label.config(text="No plots generated")
            return
        
        # Clear existing
        for widget in self.adv_plot_frame.winfo_children():
            widget.destroy()
        
        try:
            plot_names = list(self.adv_generated_plots.keys())
            plot_path = self.adv_generated_plots[plot_names[self.adv_current_plot_index]]
            
            from PIL import Image, ImageTk
            img = Image.open(plot_path)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = ttk.Label(self.adv_plot_frame, image=photo)
            label.image = photo
            label.pack(expand=True)
            
            self.adv_nav_label.config(
                text=f"Plot {self.adv_current_plot_index + 1} of {len(self.adv_generated_plots)}: {plot_names[self.adv_current_plot_index]}"
            )
            
        except Exception as e:
            error_label = ttk.Label(self.adv_plot_frame, text=f"Error: {str(e)}")
            error_label.pack(expand=True)

    def _adv_previous_plot(self):
        """Show previous plot in advanced interface"""
        if self.adv_generated_plots and self.adv_current_plot_index > 0:
            self.adv_current_plot_index -= 1
            self._adv_display_current_plot()

    def _adv_next_plot(self):
        """Show next plot in advanced interface"""
        if self.adv_generated_plots and self.adv_current_plot_index < len(self.adv_generated_plots) - 1:
            self.adv_current_plot_index += 1
            self._adv_display_current_plot()

    def _adv_save_current_plot(self):
        """Save current plot from advanced interface"""
        if not self.adv_generated_plots:
            return
        
        plot_names = list(self.adv_generated_plots.keys())
        current_plot = plot_names[self.adv_current_plot_index]
        current_path = self.adv_generated_plots[current_plot]
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title=f"Save {current_plot}",
            initialfile=f"{current_plot}.png"
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy2(current_path, save_path)
                messagebox.showinfo("Success", f"Saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")

    def _adv_open_plot_folder(self):
        """Open advanced plots folder"""
        plot_dir = os.path.join(self.OUTPUT_DIR, "advanced_plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        
        try:
            if os.name == "nt":
                os.startfile(plot_dir)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", plot_dir])
        except:
            messagebox.showinfo("Plot Folder", f"Plots saved in:\n{plot_dir}")

    def _refresh_gallery(self):
        """Refresh the plot gallery"""
        if hasattr(self, 'gallery_list'):
            self.gallery_list.delete(0, tk.END)
            
            # Combine all generated plots
            all_plots = {}
            if hasattr(self, 'adv_generated_plots'):
                all_plots.update(self.adv_generated_plots)
            if hasattr(self, 'generated_plots'):
                all_plots.update(self.generated_plots)
            
            for plot_name in all_plots.keys():
                self.gallery_list.insert(tk.END, plot_name)

    def _on_gallery_double_click(self, event):
        """Handle gallery double-click"""
        if hasattr(self, 'gallery_list'):
            selection = self.gallery_list.curselection()
            if selection:
                plot_name = self.gallery_list.get(selection[0])
                
                # Find the plot in generated plots
                all_plots = {}
                if hasattr(self, 'adv_generated_plots'):
                    all_plots.update(self.adv_generated_plots)
                if hasattr(self, 'generated_plots'):
                    all_plots.update(self.generated_plots)
                
                if plot_name in all_plots:
                    # Switch to advanced interface and display this plot
                    if hasattr(self, 'adv_generated_plots') and plot_name in self.adv_generated_plots:
                        self.adv_current_plot_index = list(self.adv_generated_plots.keys()).index(plot_name)
                        self._adv_display_current_plot()

    def _create_basic_plots_fallback(self, output_dir):
        """Create basic plots when EnhancedVisualizer is not available"""
        try:
            self.generated_plots = {}
            
            # Basic metrics plot
            metrics = self.last_pycaret_results.get('metrics', {})
            if metrics:
                plt.figure(figsize=(10, 6))
                metric_names = []
                metric_values = []
                
                for metric, value in metrics.items():
                    if metric not in ['run_log', 'test_predictions_made', 'test_set_size']:
                        metric_names.append(metric)
                        metric_values.append(float(value))
                
                if metric_names:
                    plt.bar(metric_names, metric_values)
                    plt.title('Model Performance Metrics')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(output_dir, "metrics_basic.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.generated_plots['Metrics'] = plot_path
            
            # Feature importance if available
            model = self.last_pycaret_results.get('model')
            if hasattr(model, 'feature_importances_'):
                features = self.last_pycaret_results.get('features', [])
                importances = model.feature_importances_
                
                plt.figure(figsize=(12, 8))
                indices = np.argsort(importances)[::-1]
                plt.bar(range(len(importances)), importances[indices])
                plt.title('Feature Importances')
                if features:
                    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
                plt.tight_layout()
                
                plot_path = os.path.join(output_dir, "feature_importance_basic.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.generated_plots['Feature Importance'] = plot_path
            
            if self.generated_plots:
                self.current_plot_index = 0
                self._display_current_plot()
                self.status.config(text=f"Generated {len(self.generated_plots)} basic plots")
            else:
                self.status.config(text="Could not generate any plots")
                
        except Exception as e:
            self._log_ml_message(f"❌ Basic plot fallback failed: {e}")

    def _generate_single_feature_plot(self):
        """Generate single feature vs target plot"""
        if not hasattr(self, 'last_pycaret_results'):
            messagebox.showinfo("No Results", "No training results available.")
            return
        
        feature_name = self.feature_selection.get()
        if not feature_name:
            messagebox.showinfo("No Feature", "Please select a feature first.")
            return
        
        try:
            # Create output directory
            output_dir = os.path.join(getattr(self, 'OUTPUT_DIR', OUTPUT_DIR), "results_plots")
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if EnhancedVisualizer is available
            if EnhancedVisualizer is None:
                messagebox.showerror("Error", "EnhancedVisualizer not available.")
                return
            
            # Prepare data for visualization
            test_data_for_viz = None
            train_data_for_viz = None
            
            # Get test data
            if hasattr(self, 'test_data') and self.test_data[0] is not None:
                features = self.last_pycaret_results.get('features', [])
                if features and len(features) > 0:
                    try:
                        if isinstance(self.test_data[0], pd.DataFrame):
                            test_data_for_viz = (self.test_data[0][features], self.y_true)
                        else:
                            test_data_for_viz = (self.test_data[0], self.y_true)
                    except Exception as e:
                        print(f"Test data preparation error: {e}")
            
            # Get train data
            train_data_for_viz = self.last_pycaret_results.get('train_data')
            if train_data_for_viz is None and hasattr(self, 'train_data'):
                train_data_for_viz = self.train_data
            
            # Get features and target
            features = self.last_pycaret_results.get('features', [])
            target = self.last_pycaret_results.get('target')
            
            # Create the enhanced visualizer
            visualizer = EnhancedVisualizer(
                self.last_pycaret_results,
                self.last_pycaret_results.get('problem_type', 'regression'),
                self.last_pycaret_results.get('model'),
                test_data_for_viz,
                train_data_for_viz,
                features,
                target
            )
            
            # Generate single feature plot
            plot_path = visualizer.create_dynamic_comparison(output_dir, feature_name)
            
            if plot_path:
                # Add to generated plots and display
                plot_name = f"Feature Analysis: {feature_name}"
                self.generated_plots[plot_name] = plot_path
                self.current_plot_index = len(self.generated_plots) - 1
                self._display_current_plot()
                self._log_ml_message(f"✅ Generated: {plot_name}")
                
                # Update navigation label
                if hasattr(self, 'plot_nav_label'):
                    self.plot_nav_label.config(
                        text=f"Plot {self.current_plot_index + 1} of {len(self.generated_plots)}: {plot_name}"
                    )
            else:
                messagebox.showinfo("Error", f"Could not generate plot for {feature_name}")
                
        except Exception as e:
            error_msg = f"Failed to generate single feature plot: {str(e)}"
            self._log_ml_message(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)

    def _save_current_plot(self):
        """Save the current plot to a file"""
        if not self.generated_plots or self.current_plot_index >= len(self.generated_plots):
            messagebox.showinfo("No Plot", "No plot to save.")
            return
        
        try:
            plot_path = list(self.generated_plots.values())[self.current_plot_index]
            plot_name = list(self.generated_plots.keys())[self.current_plot_index]
            
            # Ask user for save location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title=f"Save {plot_name}",
                initialfile=f"{plot_name}.png"
            )
            
            if save_path:
                import shutil
                shutil.copy2(plot_path, save_path)
                messagebox.showinfo("Success", f"Plot saved to:\n{save_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plot: {str(e)}")

    def _open_interactive_view(self):
        """Open interactive plot viewer"""
        if not self.generated_plots or self.current_plot_index >= len(self.generated_plots):
            messagebox.showinfo("No Plot", "No plot to view.")
            return
        
        try:
            plot_path = list(self.generated_plots.values())[self.current_plot_index]
            plot_name = list(self.generated_plots.keys())[self.current_plot_index]
            
            # Import and open interactive viewer
            from interactive_plot_viewer import InteractivePlotViewer
            InteractivePlotViewer(self.root, plot_path)
            
        except ImportError:
            messagebox.showinfo("Feature Not Available", 
                            "Interactive plot viewer requires additional dependencies.\n"
                            "Run: pip install matplotlib")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open interactive viewer: {str(e)}")

    def cleanup(self):
        """Clean up resources and prevent threading issues"""
        try:
            # Stop any running threads
            if hasattr(self, 'training_thread') and self.training_thread.is_alive():
                self.training_thread.join(timeout=1.0)
        except:
            pass

    def _display_metrics_table(self, results):
        """Display metrics in a formatted side-by-side table (Train vs Test)"""
        if not hasattr(self, 'results_text'):
            return
            
        self.results_text.delete(1.0, "end")
        
        metrics = results.get('metrics', {})
        problem_type = results.get('problem_type', 'unknown')
        
        # Header
        self.results_text.insert("end", "=" * 65 + "\n")
        self.results_text.insert("end", f"MODEL PERFORMANCE SUMMARY ({problem_type.upper()})\n")
        self.results_text.insert("end", "=" * 65 + "\n\n")
        
        # Define metrics to show based on problem type
        if problem_type == "regression":
            metric_map = {
                'MAE': 'mae', 
                'MSE': 'mse', 
                'RMSE': 'rmse', 
                'R2 Score': 'r2', 
                'MAPE (%)': 'mape'
            }
        else: # Classification
            metric_map = {
                'Accuracy': 'accuracy', 
                'F1 Score': 'f1', 
                'Precision': 'precision', 
                'Recall': 'recall'
            }
            
        # Create Table Header
        # Format: | Metric Name | Train | Test |
        header_fmt = "{:<15} | {:>12} | {:>12}\n"
        self.results_text.insert("end", header_fmt.format("METRIC", "TRAIN", "TEST"))
        self.results_text.insert("end", "-" * 45 + "\n")
        
        # Populate Rows
        for display_name, key_suffix in metric_map.items():
            train_key = f"train_{key_suffix}"
            test_key = f"test_{key_suffix}"
            
            train_val = metrics.get(train_key, "N/A")
            test_val = metrics.get(test_key, "N/A")
            
            # Format numbers
            t_str = f"{train_val:.4f}" if isinstance(train_val, (float, int)) else str(train_val)
            test_str = f"{test_val:.4f}" if isinstance(test_val, (float, int)) else str(test_val)
            
            self.results_text.insert("end", header_fmt.format(display_name, t_str, test_str))
            
        self.results_text.insert("end", "-" * 45 + "\n\n")
        
        # Add Model Info
        model = results.get('model')
        if model:
            self.results_text.insert("end", f"📦 Model Architecture: {type(model).__name__}\n")
            
        # Add Features info
        features = results.get('features', [])
        self.results_text.insert("end", f"🔧 Features Used: {len(features)}\n")

    # Add this method to your PyCaretTrainer class
    def _safe_tkinter_callback(self, callback, *args):
        """Safely execute Tkinter callbacks in the main thread"""
        if hasattr(self, 'root') and self.root:
            self.root.after(0, lambda: callback(*args))

    # Modify your log method:
    def log(self, *parts, redirect_to_ui=True):
        """Thread-safe logging"""
        text = " ".join(str(p) for p in parts)
        self._logs.append(text)
        
        # Print to console (thread-safe)
        print(f"[PyCaret] {text}")
        
        # Thread-safe UI logging
        if redirect_to_ui and hasattr(self, 'log_queue'):
            try:
                # Use safe callback for Tkinter
                self._safe_tkinter_callback(lambda: self.log_queue.put(text))
            except:
                pass

    def _open_enhanced_preprocessor(self):
        """
        Opens the Enhanced Preprocessor using the LIVE PREVIEW data.
        Updates the main window's column list (Per-Column Settings) upon application.
        """
        # 1. Check imports
        if EnhancedPreprocessor is None:
            messagebox.showerror("Error", "EnhancedPreprocessor module failed to load.")
            return

        # 2. SOURCE OF TRUTH: Use self.preview_df
        if hasattr(self, 'preview_df') and self.preview_df is not None:
            current_df = self.preview_df
        elif hasattr(self, 'df_processed') and self.df_processed is not None:
            current_df = self.df_processed
        elif hasattr(self, 'df_original') and self.df_original is not None:
            current_df = self.df_original
        else:
            messagebox.showinfo("No Data", "Please load data first.")
            return

        # 3. Create wrapper
        class TempPreprocessorWrapper:
            def __init__(self, df):
                self.df_original = df.copy()

        temp_preprocessor = TempPreprocessorWrapper(current_df)

        # 4. Callbacks
        def on_data_update_callback(new_df):
            """Updates just the visual preview (non-destructive)"""
            self.preview_df = new_df
            self.preview_frame_preproc.set_dataframe(new_df)
            self.status.config(text="Preview updated from Advanced Tools")

        def on_final_apply_callback(new_df):
            """Updates the actual application state and REBUILDS column settings"""
            try:
                # CRITICAL: Update df_original to the new PCA data.
                # This makes the app treat the PCA output as the new "base" dataset.
                self.df_original = new_df.copy()
                self.df_processed = new_df.copy()
                
                # Re-initialize the main preprocessor with this new data
                self.preprocessor = Preprocessor(self.df_original)
                
                # REBUILD THE COLUMN SETTINGS UI (The key fix)
                # This function looks at self.df_original, which we just updated.
                # It will clear the old columns (Mileage, Price) and create controls for (PC1, PC2...)
                self._populate_preproc_columns()
                
                # Update visual previews
                self.preview_df = self.df_processed
                self.preview_frame_preproc.set_dataframe(self.df_processed)
                
                # Update ML tab dropdowns
                if hasattr(self, '_refresh_ml_columns'):
                    self._refresh_ml_columns()
                
                self.status.config(text="Advanced Transformations Applied. Column list updated.")
                return True
            except Exception as e:
                messagebox.showerror("Apply Failed", f"Error applying advanced changes: {e}\n{traceback.format_exc()}")
                return False

        # 5. Launch
        try:
            EnhancedPreprocessor(
                self.root, 
                temp_preprocessor, 
                on_data_update_callback, 
                on_final_apply_callback
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Advanced Tools: {e}\n{traceback.format_exc()}")

    def _toggle_validation_frame(self):
            """Show or hide advanced validation settings"""
            if self.show_validation_var.get():
                self.validation_frame.pack(fill="x", padx=5, pady=5, after=self.hyperparams_button)
            else:
                self.validation_frame.pack_forget()

    def _display_model_equation_or_rules(self):
        """Extracts model logic (Equation, Rules, or PCA Text) with REAL NAMES."""
        if not hasattr(self, 'last_pycaret_results'):
            messagebox.showinfo("Info", "Train a model first.")
            return

        # 1. Get Model and Info
        model_pipeline = self.last_pycaret_results.get('model')
        features = self.last_pycaret_results.get('features', []) 
        problem_type = self.last_pycaret_results.get('problem_type')
        
        if not model_pipeline: return

        # 2. Unwrap Pipeline
        from sklearn.pipeline import Pipeline
        if isinstance(model_pipeline, Pipeline):
            if 'actual_estimator' in model_pipeline.named_steps:
                model = model_pipeline.named_steps['actual_estimator']
            else:
                model = model_pipeline.steps[-1][1]
        else:
            model = model_pipeline

        # 3. Get Class Names (for Classification)
        class_names_list = None
        if problem_type == 'classification':
            target_col = self.last_pycaret_results.get('target')
            if hasattr(self, 'df_processed') and self.df_processed is not None and target_col in self.df_processed.columns:
                try:
                    unique_vals = sorted(self.df_processed[target_col].dropna().unique())
                    class_names_list = [str(x) for x in unique_vals]
                except: pass

        # 4. Generate Content Header
        model_name = type(model).__name__
        content = f"Model Architecture: {model_name}\n"
        content += f"Features Used: {len(features)}\n"
        content += "=" * 60 + "\n\n"

        try:
            # --- A. CLUSTERING (PCA Explanation) ---
            if problem_type == 'clustering':
                # Prepare data (Force inject target if missing, same as plotting)
                train_data = self.last_pycaret_results.get('train_data')
                target_col = self.last_pycaret_results.get('target')
                
                if target_col and train_data is not None:
                    if isinstance(train_data, pd.DataFrame) and target_col not in train_data.columns:
                        # Try to recover target from processed df
                        if hasattr(self, 'df_processed') and target_col in self.df_processed.columns:
                            try:
                                target_series = self.df_processed[target_col]
                                train_data = train_data.merge(target_series, left_index=True, right_index=True, how='left')
                            except: pass

                # Use Visualizer to calculate PCA text
                from enhanced_visualizer import EnhancedVisualizer
                viz = EnhancedVisualizer(
                    self.last_pycaret_results,
                    problem_type,
                    model,
                    train_data=train_data
                )
                content += viz.get_pca_explanation_text()

            # --- B. LINEAR MODELS ---
            elif hasattr(model, 'coef_'):
                content += "=== LINEAR EQUATION ===\n"
                intercept = getattr(model, 'intercept_', 0)
                if isinstance(intercept, (list, np.ndarray)): intercept = intercept[0]
                eq_parts = [f"{intercept:.4f}"]
                coefs = model.coef_
                if coefs.ndim > 1: coefs = coefs[0]
                
                for i, c in enumerate(coefs):
                    name = features[i] if i < len(features) else f"Feat_{i}"
                    sign = "+" if c >= 0 else "-"
                    eq_parts.append(f"\n   {sign} ({abs(c):.5f} * {name})")
                content += f"y = {''.join(eq_parts)}"

            # --- C. DECISION TREES / RANDOM FOREST ---
            elif hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                content += "=== DECISION RULES ===\n"
                from sklearn.tree import export_text
                est = model
                if hasattr(model, 'estimators_'): 
                    est = model.estimators_[0]
                    content += f"(Random Forest: Showing rules for Tree #1 out of {len(model.estimators_)})\n\n"
                
                try:
                    # Safe export with name matching
                    if hasattr(est, "n_features_in_") and len(features) != est.n_features_in_:
                        padded_features = features + [f"Extra_{i}" for i in range(len(features), est.n_features_in_)]
                        content += export_text(est, feature_names=padded_features, class_names=class_names_list, max_depth=10)
                    else:
                        content += export_text(est, feature_names=features, class_names=class_names_list, max_depth=10)
                except Exception as e:
                    content += f"Error mapping names: {e}\n"
                    content += export_text(est, max_depth=10)

            # --- D. BOOSTING ---
            elif 'XGB' in model_name or 'LGBM' in model_name:
                content += "=== BOOSTING RULES ===\n"
                try:
                    if hasattr(model, 'get_booster'): # XGBoost
                        dump = model.get_booster().get_dump(with_stats=True)[0]
                        for i, name in enumerate(features):
                            dump = dump.replace(f"f{i}", name)
                        content += dump
                    elif hasattr(model, 'booster_'): # LightGBM
                        content += str(model.booster_.dump_model()['tree_info'][0])
                except: 
                    content += "See Feature Importance plot."
            else:
                content += "No text rules available for this model."

        except Exception as e:
            content += f"\nError extracting logic: {str(e)}"

        # 5. Display & Save
        self.rules_text.delete("1.0", "end")
        self.rules_text.insert("1.0", content)

        try:
            output_dir = os.path.join(getattr(self, 'OUTPUT_DIR', 'output'), "results_plots")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"model_logic_{model_name}.txt"), "w", encoding="utf-8") as f:
                f.write(content)
        except: pass

    def _generate_nature_plot(self, plot_kind="tree"):
        """Generate and display the nature-defining plot (Tree or Importance)."""
        if not hasattr(self, 'last_pycaret_results'): return
        
        # Ensure output dir exists
        output_dir = os.path.join(getattr(self, 'OUTPUT_DIR', 'output'), "results_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        res = self.last_pycaret_results
        model_pipeline = res.get('model')
        
        # 1. Unwrap Pipeline just to check if it's a tree (for the if-check later)
        # We still pass the FULL pipeline to the visualizer constructor
        from sklearn.pipeline import Pipeline
        inner_model = model_pipeline
        if isinstance(model_pipeline, Pipeline):
            if 'actual_estimator' in model_pipeline.named_steps:
                inner_model = model_pipeline.named_steps['actual_estimator']
            else:
                inner_model = model_pipeline.steps[-1][1]

        # 2. Get Class Names (for Classification)
        class_names_list = None
        if res.get('problem_type') == 'classification':
            target_col = res.get('target')
            if hasattr(self, 'df_processed') and self.df_processed is not None and target_col in self.df_processed.columns:
                try:
                    unique_vals = sorted(self.df_processed[target_col].dropna().unique())
                    class_names_list = [str(x) for x in unique_vals]
                except: pass

        # 3. Prepare Data for Visualizer
        test_data_tuple = None
        if hasattr(self, 'test_data') and self.test_data:
            test_data_tuple = self.test_data

        # 4. Create Visualizer (FIXED INSTANTIATION)
        from enhanced_visualizer import EnhancedVisualizer
        viz = EnhancedVisualizer(
            results=res,
            problem_type=res.get('problem_type', 'regression'),
            model=model_pipeline,  # Pass full pipeline
            test_data=test_data_tuple,
            train_data=res.get('train_data'),
            features=res.get('features'),
            target=res.get('target')
        )
        
        plot_path = None
        
        if plot_kind == "tree":
            if hasattr(inner_model, 'tree_') or hasattr(inner_model, 'estimators_'):
                # Pass class_names to show real categories
                plot_path = viz.plot_complete_decision_tree(output_dir, max_depth=5, class_names=class_names_list)
            else:
                messagebox.showinfo("Info", "This model is not a Decision Tree or Random Forest.")
                return
        elif plot_kind == "importance":
            plot_path = viz._plot_feature_importance(output_dir)
        
        if plot_path:
            key = f"Nature_{plot_kind}"
            self.generated_plots[key] = plot_path
            self.current_plot_index = list(self.generated_plots.keys()).index(key)
            self._display_current_plot()

    def _generate_poly_analysis(self):
        """Triggers the Polynomial Analysis Plot with CUSTOM DEGREE support."""
        if not hasattr(self, 'last_pycaret_results'):
            messagebox.showinfo("Error", "Train a model first.")
            return
            
        features = self.last_pycaret_results.get('features', [])
        if not features: return
        
        # Create Popup
        top = tk.Toplevel(self.root)
        top.title("Polynomial Fit Analysis")
        top.geometry("300x250")
        
        # Feature Selection
        ttk.Label(top, text="1. Select Feature:").pack(pady=(10,0))
        combo = ttk.Combobox(top, values=features, state="readonly")
        combo.pack(pady=5)
        if features: combo.set(features[0])
        
        # Degree Selection
        ttk.Label(top, text="2. Enter Polynomial Degree:").pack(pady=(10,0))
        degree_var = tk.StringVar(value="2")
        degree_spin = ttk.Spinbox(top, from_=2, to=20, textvariable=degree_var, width=5)
        degree_spin.pack(pady=5)
        
        ttk.Label(top, text="(e.g. 2=Quadratic, 3=Cubic, 5=Complex)").pack(pady=0)
        
        def run_plot():
            feat = combo.get()
            try:
                deg = int(degree_var.get())
            except:
                deg = 2
            top.destroy()
            
            # Prepare Visualizer
            from enhanced_visualizer import EnhancedVisualizer
            output_dir = os.path.join(getattr(self, 'OUTPUT_DIR', 'output'), "results_plots")
            os.makedirs(output_dir, exist_ok=True)
            
            # Reuse existing data extraction logic
            train_data = self.last_pycaret_results.get('train_data')
            target_col = self.last_pycaret_results.get('target')
            
            # Force inject target if missing (same as clustering logic)
            if isinstance(train_data, pd.DataFrame) and target_col and target_col not in train_data.columns:
                if hasattr(self, 'df_processed') and target_col in self.df_processed.columns:
                     try:
                        target_series = self.df_processed[target_col]
                        train_data = train_data.merge(target_series, left_index=True, right_index=True, how='left')
                     except: pass

            viz = EnhancedVisualizer(
                self.last_pycaret_results,
                self.last_pycaret_results.get('problem_type'),
                self.last_pycaret_results.get('model'),
                train_data=train_data,
                features=features,
                target=target_col
            )
            
            # Pass the custom degree!
            path = viz.plot_polynomial_fit(output_dir, feat, custom_degree=deg)
            
            if path:
                self.generated_plots[f"Poly (d={deg}): {feat}"] = path
                self.current_plot_index = list(self.generated_plots.keys()).index(f"Poly (d={deg}): {feat}")
                self._display_current_plot()
                self._log_ml_message(f"✅ Generated Polynomial Plot (Degree {deg}) for {feat}")
            else:
                messagebox.showerror("Error", "Could not generate plot. (Is the feature numeric?)")

        ttk.Button(top, text="Generate Graph", command=run_plot).pack(pady=15)

class ModernSplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()  # Hide the main window immediately
        
        # Create Splash Window
        self.splash = tk.Toplevel(root)
        self.splash.overrideredirect(True)  # Remove border/title bar
        
        # Dimensions
        width = 600
        height = 350
        
        # Center on screen
        screen_width = self.splash.winfo_screenwidth()
        screen_height = self.splash.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.splash.geometry(f"{width}x{height}+{x}+{y}")
        
        # Modern Colors
        self.bg_color = "#1e1e1e"        # Dark Grey Background
        self.fg_color = "#ffffff"        # White Text
        self.accent_color = "#007acc"    # VS Code Blue
        
        self.splash.configure(bg=self.bg_color)
        
        # --- DESIGN ---
        
        # 1. Logo / Title Area
        title_font = ("Segoe UI", 24, "bold")
        subtitle_font = ("Segoe UI", 10)
        
        tk.Label(self.splash, text="QFit - Automated ML software", font=title_font, 
                 bg=self.bg_color, fg=self.fg_color).pack(pady=(60, 5))
        
        tk.Label(self.splash, text="Automated Machine Learning Environment by Syed Shayan Ahmed", 
                 font=subtitle_font, bg=self.bg_color, fg="#aaaaaa").pack(pady=(0, 40))
        
        # 2. Loading Status Text
        self.status_label = tk.Label(self.splash, text="Initializing...", 
                                   font=("Segoe UI", 9), bg=self.bg_color, fg="#cccccc")
        self.status_label.pack(side="bottom", pady=(0, 20))
        
        # 3. Custom Progress Bar (Canvas for smoother look)
        self.progress_frame = tk.Frame(self.splash, bg=self.bg_color, height=4, width=400)
        self.progress_frame.pack(side="bottom", pady=(0, 10))
        
        self.canvas = tk.Canvas(self.progress_frame, width=400, height=4, 
                              bg="#333333", highlightthickness=0)
        self.canvas.pack()
        self.bar = self.canvas.create_rectangle(0, 0, 0, 4, fill=self.accent_color, width=0)
        
        self.splash.update()

    def update(self, percent, message):
        """Update progress bar and message"""
        # Update Text
        self.status_label.config(text=message)
        
        # Update Bar Width (Smooth animation calculation)
        new_width = 400 * (percent / 100)
        self.canvas.coords(self.bar, 0, 0, new_width, 4)
        
        self.splash.update()
        
    def close(self):
        """Destroy splash and show main window"""
        self.splash.destroy()
        self.root.deiconify()  # Show main window
            
if __name__ == "__main__":
    import time
    
    # 1. Create Root (Hidden)
    multiprocessing.freeze_support()
    root = tk.Tk()
    
    # 2. Launch Splash Screen
    splash = ModernSplashScreen(root)
    
    # 3. Simulate Loading Steps (The "Beautiful" Part)
    loading_steps = [
        (5,  "Initializing User Interface..."),
        (15, "Loading Data Processor Modules..."),
        (25, "Configuring Pandas & NumPy Kernels..."),
        (40, "Connecting to PyCaret Engine..."),
        (55, "Loading Visualization Libraries (Matplotlib/Seaborn)..."),
        (70, "Optimizing Runtime Environment..."),
        (85, "Preparing Analysis Dashboard..."),
        (95, "Finalizing Setup..."),
        (100, "Ready!")
    ]
    
    for percent, message in loading_steps:
        splash.update(percent, message)
        # Adjust speed here (0.1 is fast, 0.3 is cinematic)
        time.sleep(1) 
        
    # 4. Close Splash and Start App
    splash.close()
    
    app = App(root)
    try:
        root.mainloop()
    finally:
        app.cleanup()