
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import math
import numpy as np

# Try to import AdvancedPreprocessor to create temporary preprocessors for accurate previews.
# If not available at runtime, the editor will fallback to using the provided preprocessor instance.
try:
    from preprocess import AdvancedPreprocessor
except Exception:
    AdvancedPreprocessor = None

class AdvancedEditor:


    def __init__(self, parent, preprocessor, column, current_settings, on_settings_change):
        self.parent = parent
        self.preprocessor = preprocessor
        self.column = column
        self.on_settings_change = on_settings_change
        # copy settings to avoid mutating caller dict
        self.current_settings = dict(current_settings) if current_settings else {}

        # orientation
        self.orientation = self.current_settings.get("orientation", "horizontal")

        # default keys
        defaults = {
            "strategy": self.current_settings.get("strategy", "none"),
            "fill_value": self.current_settings.get("fill_value", ""),
            "low": self.current_settings.get("low", ""),
            "high": self.current_settings.get("high", ""),
            "onehot": self.current_settings.get("onehot", False),
            "label": self.current_settings.get("label", False),
            "custom_mapping": self.current_settings.get("custom_mapping", ""),
            "bins": self.current_settings.get("bins", ""),
            "bin_labels": self.current_settings.get("bin_labels", ""),
            "pattern_groups": self.current_settings.get("pattern_groups", ""),
            "remove_values": self.current_settings.get("remove_values", ""),
            "remove_mode": self.current_settings.get("remove_mode", "equals"),
            "clamp_min": self.current_settings.get("clamp_min", ""),
            "clamp_min_to": self.current_settings.get("clamp_min_to", ""),
            "clamp_max": self.current_settings.get("clamp_max", ""),
            "clamp_max_to": self.current_settings.get("clamp_max_to", ""),
            "percent_rules": self.current_settings.get("percent_rules", ""),
            "strip_substring": self.current_settings.get("strip_substring", "")
        }
        self.current_settings = defaults

        # state
        self._updating = False

        # widgets that will be created
        self.dialog = None
        self.tree_original = None
        self.tree_processed = None

        self._create_ui()

    def _create_ui(self):
        try:
            if self.dialog and self.dialog.winfo_exists():
                self.dialog.destroy()
        except Exception:
            pass

        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(f"Advanced Editor — {self.column}")
        self.dialog.geometry("1200x750")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Top toolbar
        topbar = ttk.Frame(self.dialog)
        topbar.pack(fill="x", padx=6, pady=4)
        ttk.Button(topbar, text="Apply All", style="Accent.TButton",command=self._apply_all_settings).pack(side="left", padx=4)
        ttk.Button(topbar, text="Remove All", style="Accent.TButton",command=self._clear_all_settings).pack(side="left", padx=4)
        ttk.Button(topbar, text="Refresh Preview", command=self._force_refresh).pack(side="left", padx=4)
        ttk.Button(topbar, text="Switch Layout", command=self._switch_layout).pack(side="left", padx=4)
        ttk.Button(topbar, text="Close", command=self.dialog.destroy).pack(side="right", padx=4)

        # Main panes
        orient = tk.VERTICAL if self.orientation == "vertical" else tk.HORIZONTAL
        main_paned = ttk.Panedwindow(self.dialog, orient=orient)
        main_paned.pack(fill="both", expand=True, padx=6, pady=6)

        controls_frame = ttk.Frame(main_paned)
        self._create_controls(controls_frame)
        main_paned.add(controls_frame, weight=1)

        preview_frame = ttk.Frame(main_paned)
        self._create_preview(preview_frame)
        main_paned.add(preview_frame, weight=2)

        # Delay first preview to ensure widgets exist
        self.dialog.after(80, self._on_setting_change)

    def _create_controls(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        basic_tab = ttk.Frame(nb); self._populate_basic_tab(basic_tab); nb.add(basic_tab, text="Basic")
        mapping_tab = ttk.Frame(nb); self._populate_mapping_tab(mapping_tab); nb.add(mapping_tab, text="Value Mapping")
        binning_tab = ttk.Frame(nb); self._populate_binning_tab(binning_tab); nb.add(binning_tab, text="Binning")
        pattern_tab = ttk.Frame(nb); self._populate_pattern_tab(pattern_tab); nb.add(pattern_tab, text="Patterns")
        remove_tab = ttk.Frame(nb); self._populate_remove_tab(remove_tab); nb.add(remove_tab, text="Row Filters")
        clamp_tab = ttk.Frame(nb); self._populate_clamp_tab(clamp_tab); nb.add(clamp_tab, text="Clamping")
        percent_tab = ttk.Frame(nb); self._populate_percentage_tab(percent_tab); nb.add(percent_tab, text="Percentage Rules")

    def _populate_basic_tab(self, parent):
        # Create a scrollable frame for the basic tab
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Now create the content in the scrollable frame
        frm = ttk.LabelFrame(scrollable_frame, text="Missing Values & Encoding")
        frm.pack(fill="x", padx=6, pady=6)
        
        # Use grid with proper row management
        row = 0
        
        # Missing values strategy
        ttk.Label(frm, text="Strategy:").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        self.missing_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("strategy","none"))
        cb = ttk.Combobox(frm, textvariable=self.missing_var, 
                        values=["none","mean","median","mode","fill_value","drop_row"], 
                        state="readonly", width=14)
        cb.grid(row=row, column=1, padx=4, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_setting_change())
        row += 1
        
        # Fill value
        ttk.Label(frm, text="Fill Value:").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        self.fill_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("fill_value",""))
        e = ttk.Entry(frm, textvariable=self.fill_var, width=20)
        e.grid(row=row, column=1, padx=4, pady=2)
        e.bind("<KeyRelease>", lambda e: self._on_setting_change())
        row += 1

        # ⭐⭐ DATA TYPE CONVERSION - Make sure this is visible ⭐⭐
        ttk.Label(frm, text="Convert to:").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        self.dtype_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("dtype_conversion",""))
        dtype_combo = ttk.Combobox(frm, textvariable=self.dtype_var, 
                                values=["", "integer", "float", "boolean", "string", "category"], 
                                state="readonly", width=14)
        dtype_combo.grid(row=row, column=1, padx=4, pady=2)
        dtype_combo.bind("<<ComboboxSelected>>", lambda e: self._on_setting_change())
        row += 1

        # Outlier removal
        out_frm = ttk.LabelFrame(scrollable_frame, text="Outlier Removal (drop rows outside range)")
        out_frm.pack(fill="x", padx=6, pady=6)
        
        ttk.Label(out_frm, text="Min Value:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.min_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("low",""))
        e1 = ttk.Entry(out_frm, textvariable=self.min_var, width=20)
        e1.grid(row=0, column=1, padx=4, pady=2)
        e1.bind("<KeyRelease>", lambda e: self._on_setting_change())
        
        ttk.Label(out_frm, text="Max Value:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.max_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("high",""))
        e2 = ttk.Entry(out_frm, textvariable=self.max_var, width=20)
        e2.grid(row=1, column=1, padx=4, pady=2)
        e2.bind("<KeyRelease>", lambda e: self._on_setting_change())

        # Encoding
        enc_frm = ttk.LabelFrame(scrollable_frame, text="Encoding")
        enc_frm.pack(fill="x", padx=6, pady=6)
        
        self.onehot_var = tk.BooleanVar(master=self.dialog, value=self.current_settings.get("onehot",False))
        ttk.Checkbutton(enc_frm, text="One-Hot Encoding", variable=self.onehot_var, 
                    command=self._on_setting_change).pack(anchor="w")
        
        self.label_var = tk.BooleanVar(master=self.dialog, value=self.current_settings.get("label",False))
        ttk.Checkbutton(enc_frm, text="Label Encoding", variable=self.label_var, 
                    command=self._on_setting_change).pack(anchor="w")

        # Strip substring
        strip_frame = ttk.LabelFrame(scrollable_frame, text="Strip Substring (Remove text from all values)")
        strip_frame.pack(fill="x", padx=6, pady=6)

        self.strip_substring_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("strip_substring", ""))
        ttk.Label(strip_frame, text="Substring to remove:").pack(side="left", padx=4)
        entry_strip = ttk.Entry(strip_frame, textvariable=self.strip_substring_var, width=25)
        entry_strip.pack(side="left", padx=4)
        entry_strip.bind("<KeyRelease>", lambda e: self._on_setting_change())

        range_frame = ttk.LabelFrame(scrollable_frame, text="Range-based Classification")
        range_frame.pack(fill="x", padx=6, pady=6)

        ttk.Label(range_frame, text="Range Rules (format: min-max:label):").pack(anchor="w", padx=4, pady=2)
        self.range_mapping_var = tk.StringVar(value=self.current_settings.get("range_mapping", ""))
        range_entry = ttk.Entry(range_frame, textvariable=self.range_mapping_var, width=60)
        range_entry.pack(fill="x", padx=4, pady=2)
        range_entry.bind("<KeyRelease>", lambda e: self._on_setting_change())

        # Add example label
        example_label = ttk.Label(range_frame, text="Example: 0-3:A, 4-7:B, 8-10:C", 
                                font=("", 8), foreground="gray")
        example_label.pack(anchor="w", padx=4, pady=2)
    
    def _populate_mapping_tab(self, parent):
        main = ttk.Frame(parent); main.pack(fill="both",expand=True,padx=6,pady=6)
        assign_frame = ttk.LabelFrame(main, text="Quick Value Assignment"); assign_frame.pack(fill="x",padx=4,pady=4)
        ttk.Label(assign_frame, text="Assign to:").pack(side="left", padx=4)
        self.assign_var = tk.StringVar(master=self.dialog, value=""); ttk.Entry(assign_frame, textvariable=self.assign_var, width=15).pack(side="left", padx=4)
        ttk.Button(assign_frame, text="Assign Selected", command=self._assign_selected_values).pack(side="left", padx=4)
        ttk.Button(assign_frame, text="Select All", command=lambda: self._set_all_value_vars(True)).pack(side="left", padx=4)
        ttk.Button(assign_frame, text="Clear Selection", command=lambda: self._set_all_value_vars(False)).pack(side="left", padx=4)
        quick_lbls = ["A","B","C","low","medium","high","yes","no"]
        qf = ttk.Frame(assign_frame); qf.pack(fill="x", pady=4)
        for lbl in quick_lbls: ttk.Button(qf, text=lbl, width=6, command=lambda l=lbl: self.assign_var.set(l)).pack(side="left", padx=2)

        values_frame = ttk.LabelFrame(main, text="Top Values (select to assign)"); values_frame.pack(fill="both",expand=True,padx=4,pady=4)
        canvas = tk.Canvas(values_frame, height=260); scrollbar = ttk.Scrollbar(values_frame, orient="vertical", command=canvas.yview); inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set); canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")

        header = ttk.Frame(inner); header.pack(fill="x", pady=2)
        ttk.Label(header, text="Value", width=40).pack(side="left"); ttk.Label(header, text="Count", width=8).pack(side="left"); ttk.Label(header, text="%", width=6).pack(side="left"); ttk.Label(header, text="Sel", width=6).pack(side="left")

        series = self.preprocessor.df_original[self.column]
        counts = series.value_counts(dropna=False).head(200)
        total_non_na = len(series.dropna())
        self.value_vars = {}
        for val,cnt in counts.items():
            row = ttk.Frame(inner); row.pack(fill="x", pady=1)
            vstr = str(val) if pd.notna(val) else "NaN"
            display = vstr if len(vstr) <= 40 else vstr[:37] + "..."
            ttk.Label(row, text=display, width=40, anchor="w").pack(side="left")
            ttk.Label(row, text=str(cnt), width=8).pack(side="left")
            pct = (cnt/total_non_na)*100 if total_non_na>0 else 0.0
            ttk.Label(row, text=f"{pct:.1f}%", width=6).pack(side="left")
            var = tk.BooleanVar(master=self.dialog, value=False)
            chk = ttk.Checkbutton(row, variable=var); chk.pack(side="left", padx=6)
            self.value_vars[vstr] = var

        log_frame = ttk.LabelFrame(main, text="Transformation Logs")
        log_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.mapping_log_text = tk.Text(log_frame, height=10)
        self.mapping_log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Automatically refresh logs when settings change
        def update_logs_local():
            logs = self.preprocessor.get_transformations_log() or []
            self.mapping_log_text.config(state="normal")
            self.mapping_log_text.delete("1.0", "end")
            for l in logs:
                self.mapping_log_text.insert("end", f"• {l}\n")
            self.mapping_log_text.config(state="disabled")

        # link the updater
        self._update_mapping_logs = update_logs_local


    def _populate_binning_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Binning"); frm.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Label(frm, text="Bins (comma separated numeric boundaries):").pack(anchor="w", padx=4, pady=2)
        self.bins_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("bins",""))
        ttk.Entry(frm, textvariable=self.bins_var, width=60).pack(fill="x", padx=4, pady=2)
        ttk.Label(frm, text="Bin labels (comma separated) — must be len(bins)-1").pack(anchor="w", padx=4, pady=2)
        self.bin_labels_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("bin_labels",""))
        ttk.Entry(frm, textvariable=self.bin_labels_var, width=60).pack(fill="x", padx=4, pady=2)
        ttk.Button(frm, text="Apply Binning (preview)", command=self._on_setting_change).pack(pady=6)

    def _populate_pattern_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Pattern Groups (regex:label)"); frm.pack(fill="both", expand=True, padx=6, pady=6)
        self.patterns_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("pattern_groups",""))
        ttk.Entry(frm, textvariable=self.patterns_var, width=80).pack(fill="x", padx=4, pady=4)
        ttk.Button(frm, text="Apply Patterns (preview)", command=self._on_setting_change).pack(pady=6)

    def _populate_remove_tab(self, parent):
        main = ttk.Frame(parent)
        main.pack(fill="both", expand=True, padx=6, pady=6)
        
        # ⭐ ADD THIS LINE HERE ⭐
        self.remove_values_var = tk.StringVar(
            master=self.dialog,
            value=self.current_settings.get("remove_values", "")
        )
        # Quick selection buttons
        assign_frame = ttk.LabelFrame(main, text="Quick Selection")
        assign_frame.pack(fill="x", padx=4, pady=4)
        
        ttk.Button(assign_frame, text="Select All", command=lambda: self._set_all_remove_vars(True)).pack(side="left", padx=4)
        ttk.Button(assign_frame, text="Clear Selection", command=lambda: self._set_all_remove_vars(False)).pack(side="left", padx=4)
        
        # Values list
        values_frame = ttk.LabelFrame(main, text="Select Values to Remove (changes update main preview)")
        values_frame.pack(fill="both", expand=True, padx=4, pady=4)
        
        canvas = tk.Canvas(values_frame, height=300)  # Increased height
        scrollbar = ttk.Scrollbar(values_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header
        header = ttk.Frame(inner)
        header.pack(fill="x", pady=2)
        ttk.Label(header, text="Value", width=40).pack(side="left")
        ttk.Label(header, text="Count", width=8).pack(side="left")
        ttk.Label(header, text="%", width=6).pack(side="left")
        ttk.Label(header, text="Remove", width=6).pack(side="left")

        # Populate values
        series = self.preprocessor.df_original[self.column]
        counts = series.value_counts(dropna=False).head(200)
        total_non_na = len(series.dropna())
        self.remove_value_vars = {}
        
        # Get current remove values to pre-select checkboxes
        current_remove = self.current_settings.get("remove_values", "").strip()
        current_remove_list = [v.strip() for v in current_remove.split(",") if v.strip()] if current_remove else []
        
        for val, cnt in counts.items():
            row = ttk.Frame(inner)
            row.pack(fill="x", pady=1)
            
            vstr = str(val) if pd.notna(val) else "NaN"
            display = vstr if len(vstr) <= 40 else vstr[:37] + "..."
            
            ttk.Label(row, text=display, width=40, anchor="w").pack(side="left")
            ttk.Label(row, text=str(cnt), width=8).pack(side="left")
            
            pct = (cnt / total_non_na) * 100 if total_non_na > 0 else 0.0
            ttk.Label(row, text=f"{pct:.1f}%", width=6).pack(side="left")
            
            # Pre-select if this value is in current remove list
            is_selected = vstr in current_remove_list
            var = tk.BooleanVar(master=self.dialog, value=is_selected)
            chk = ttk.Checkbutton(row, variable=var, command=self._on_remove_selection_change)
            chk.pack(side="left", padx=6)
            self.remove_value_vars[vstr] = var

        # Removal mode selection
        mode_frame = ttk.LabelFrame(main, text="Removal Settings")
        mode_frame.pack(fill="x", padx=4, pady=4)
        
        ttk.Label(mode_frame, text="Removal Mode:").pack(side="left", padx=4)
        self.remove_mode_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("remove_mode", "equals"))
        cb = ttk.Combobox(mode_frame, textvariable=self.remove_mode_var, 
                        values=["equals", "contains", "startswith", "endswith", "regex"], 
                        state="readonly", width=12)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_setting_change())
        
        # Auto-remove rare values
        auto_frame = ttk.LabelFrame(main, text="Auto-Remove Rare Values")
        auto_frame.pack(fill="x", padx=4, pady=4)
        
        ttk.Label(auto_frame, text="Remove values with frequency <").pack(side="left")
        self.remove_below_percent_var = tk.StringVar(value="")
        percent_entry = ttk.Entry(auto_frame, textvariable=self.remove_below_percent_var, width=8)
        percent_entry.pack(side="left", padx=4)
        ttk.Label(auto_frame, text="%").pack(side="left")
        ttk.Button(auto_frame, text="Find & Select", 
                command=lambda: self._apply_percentage_removal()).pack(side="left", padx=8)

    def _apply_percentage_removal(self):
        """Remove values that occur below specified percentage threshold - FIXED VERSION"""
        try:
            percent_str = self.remove_below_percent_var.get().strip()
            if not percent_str:
                return
                
            threshold = float(percent_str)
            if threshold <= 0 or threshold >= 100:
                messagebox.showerror("Invalid threshold", "Enter percentage between 0 and 100")
                return
                
            series = self.preprocessor.df_original[self.column]
            value_counts = series.value_counts(normalize=True) * 100
            
            # Find values below threshold
            rare_values = value_counts[value_counts < threshold].index.tolist()
            
            if rare_values:
                # Get current selections
                current_remove = self.remove_values_var.get().strip()
                current_list = [v.strip() for v in current_remove.split(",") if v.strip()] if current_remove else []
                
                # Add rare values that aren't already in the list
                new_values = []
                for val in rare_values:
                    val_str = str(val)
                    if val_str not in current_list:
                        new_values.append(val_str)
                
                if new_values:
                    # Update the entry field
                    updated_list = current_list + new_values
                    self.remove_values_var.set(", ".join(updated_list))
                    
                    # Also update the checkboxes
                    for val_str in new_values:
                        if val_str in self.remove_value_vars:
                            self.remove_value_vars[val_str].set(True)
                    
                    messagebox.showinfo("Rare Values Found", 
                                    f"Added {len(new_values)} rare values occurring < {threshold}%")
                    self._on_setting_change()
                else:
                    messagebox.showinfo("No New Rare Values", f"All rare values already in removal list")
            else:
                messagebox.showinfo("No Rare Values", f"No values found below {threshold}%")
                
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a valid percentage number")

    def _populate_clamp_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Clamp Numeric Values (replace out-of-range values)"); frm.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Label(frm, text="If value < (min):").grid(row=0,column=0,sticky="w",padx=4,pady=2)
        self.clamp_min_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("clamp_min",""))
        ttk.Entry(frm, textvariable=self.clamp_min_var, width=20).grid(row=0,column=1,padx=4,pady=2)
        ttk.Label(frm, text="Replace with:").grid(row=0,column=2,sticky="w",padx=4,pady=2)
        self.clamp_min_to_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("clamp_min_to",""))
        ttk.Entry(frm, textvariable=self.clamp_min_to_var, width=20).grid(row=0,column=3,padx=4,pady=2)
        ttk.Label(frm, text="If value > (max):").grid(row=1,column=0,sticky="w",padx=4,pady=2)
        self.clamp_max_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("clamp_max",""))
        ttk.Entry(frm, textvariable=self.clamp_max_var, width=20).grid(row=1,column=1,padx=4,pady=2)
        ttk.Label(frm, text="Replace with:").grid(row=1,column=2,sticky="w",padx=4,pady=2)
        self.clamp_max_to_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("clamp_max_to",""))
        ttk.Entry(frm, textvariable=self.clamp_max_to_var, width=20).grid(row=1,column=3,padx=4,pady=2)
        ttk.Button(frm, text="Apply Clamping (preview)", command=self._on_setting_change).grid(row=2,column=0,columnspan=4,pady=8)

    def _populate_percentage_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Assign Categories by Percentage (top -> down)"); frm.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Label(frm, text="Rules format: percentage:label, percentage:label, ...").pack(anchor="w", padx=4, pady=2)
        ttk.Label(frm, text="Example: 10:High,40:Medium,50:Low").pack(anchor="w", padx=4, pady=0)
        self.percent_rules_var = tk.StringVar(master=self.dialog, value=self.current_settings.get("percent_rules",""))
        ttk.Entry(frm, textvariable=self.percent_rules_var, width=80).pack(fill="x", padx=4, pady=6)
        ttk.Button(frm, text="Apply Percentage Rules (preview)", command=self._on_setting_change).pack(pady=6)

    def _create_preview(self, parent):
        # Remove the original/processed preview and just show detailed logs
        logf = ttk.LabelFrame(parent, text="Transformation Details & Logs")
        logf.pack(fill="both", expand=True, padx=6, pady=6)
        
        # Detailed transformation info
        info_frame = ttk.Frame(logf)
        info_frame.pack(fill="x", padx=4, pady=4)
        
        ttk.Label(info_frame, text="Column:", font=("", 9, "bold")).pack(side="left", padx=2)
        ttk.Label(info_frame, text=self.column, font=("", 9)).pack(side="left", padx=5)
        
        # Show column stats
        try:
            stats = self.preprocessor.get_column_stats(self.column)
            if stats:
                stats_text = f"Rows: {stats['total_rows']} | Missing: {stats['missing_values']} | Unique: {stats['unique_values']}"
                if 'dtype' in stats:
                    stats_text += f" | Type: {stats['dtype']}"
                ttk.Label(info_frame, text=stats_text, font=("", 8)).pack(side="right", padx=5)
        except:
            pass
        
        # Log area - make it larger since we removed the preview
        self.log_text = tk.Text(logf, height=15)
        scrollbar = ttk.Scrollbar(logf, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        scrollbar.pack(side="right", fill="y", padx=4, pady=4)
        self.log_text.config(state="disabled")
        
        # Add note about preview
        note_frame = ttk.Frame(logf)
        note_frame.pack(fill="x", padx=4, pady=2)
        ttk.Label(note_frame, text="💡 Changes are reflected in the main preprocessing preview", 
                font=("", 8), foreground="blue").pack()

    # -------------------- Helpers / Actions --------------------
    def _mapping_text_changed(self):
        txt = self.mapping_text.get("1.0","end-1c").strip()
        self.current_settings["custom_mapping"] = txt
        self._on_setting_change()

    def _assign_selected_values(self):
        label = self.assign_var.get().strip()
        if not label:
            messagebox.showwarning("No label","Enter a label to assign")
            return
        selected = [v for v,var in self.value_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("No selection","Select values to assign")
            return
        existing = self.current_settings.get("custom_mapping","").strip()
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        sel_set = set(selected)
        new_parts = []
        for p in parts:
            if ":" in p:
                orig = p.split(":",1)[0].strip()
                if orig in sel_set:
                    continue
            new_parts.append(p)
        for orig in selected:
            new_parts.append(f"{orig}:{label}")
        self.current_settings["custom_mapping"] = ",".join(new_parts)
        for v in self.value_vars.values(): v.set(False)
        messagebox.showinfo("Assigned", f"Assigned {len(selected)} values to '{label}'")
        self._on_setting_change()

    def _set_all_value_vars(self, val:bool):
        for v in self.value_vars.values(): v.set(val)

    def _switch_layout(self):
        self.orientation = "vertical" if self.orientation=="horizontal" else "horizontal"
        self.current_settings["orientation"] = self.orientation
        saved = dict(self.current_settings)
        self._create_ui()
        self.current_settings.update(saved)

    def _force_refresh(self):
        self._on_setting_change()

    def _collect_settings(self):
        def safe_get(varname, fallback):
            attr = getattr(self, varname, None)
            if attr is None:
                return self.current_settings.get(varname.replace("_var",""), fallback)
            try:
                return attr.get()
            except Exception:
                return self.current_settings.get(varname.replace("_var",""), fallback)
        settings = {
            "strategy": safe_get("missing_var", self.current_settings.get("strategy","none")),
            "fill_value": safe_get("fill_var", self.current_settings.get("fill_value","")),
            "low": safe_get("min_var", self.current_settings.get("low","")),
            "high": safe_get("max_var", self.current_settings.get("high","")),
            "onehot": self.onehot_var.get() if hasattr(self,"onehot_var") else self.current_settings.get("onehot",False),
            "label": self.label_var.get() if hasattr(self,"label_var") else self.current_settings.get("label",False),
            "custom_mapping": self.current_settings.get("custom_mapping",""),
            "bins": safe_get("bins_var", self.current_settings.get("bins","")),
            "range_mapping": self.range_mapping_var.get() if hasattr(self, "range_mapping_var") else self.current_settings.get("range_mapping", ""),
            "bin_labels": safe_get("bin_labels_var", self.current_settings.get("bin_labels","")),
            "pattern_groups": safe_get("patterns_var", self.current_settings.get("pattern_groups","")),
            "remove_values": safe_get("remove_values_var", self.current_settings.get("remove_values","")),
            "remove_mode": safe_get("remove_mode_var", self.current_settings.get("remove_mode","equals")),
            "clamp_min": safe_get("clamp_min_var", self.current_settings.get("clamp_min","")),
            "clamp_min_to": safe_get("clamp_min_to_var", self.current_settings.get("clamp_min_to","")),
            "clamp_max": safe_get("clamp_max_var", self.current_settings.get("clamp_max","")),
            "clamp_max_to": safe_get("clamp_max_to_var", self.current_settings.get("clamp_max_to","")),
            "percent_rules": safe_get("percent_rules_var", self.current_settings.get("percent_rules","")),
            "strip_substring": safe_get("strip_substring_var", self.current_settings.get("strip_substring","")),
            "dtype_conversion": self.dtype_var.get() if hasattr(self, "dtype_var") else self.current_settings.get("dtype_conversion", "")
        }
        # normalize booleans
        for b in ("onehot","label"):
            if isinstance(settings.get(b), str):
                settings[b] = settings[b].lower() in ("1","true","yes","y","t")
        self.current_settings.update(settings)
        return settings

    def _apply_local_row_filter(self, df:pd.DataFrame, settings:dict):
        raw = settings.get("remove_values","").strip()
        mode = settings.get("remove_mode","equals")
        if not raw:
            return df
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        if not vals:
            return df
        col = self.column
        series = df[col].astype(str).fillna("")
        # create cleaned version for comparison
        cleaned = series.str.strip().str.lower()
        keep_mask = pd.Series(True, index=df.index)
        for v in vals:
            v_clean = v.strip().lower()
            if mode == "equals":
                keep_mask &= cleaned != v_clean
            elif mode == "contains":
                keep_mask &= ~cleaned.str.contains(v_clean, na=False)
            elif mode == "startswith":
                keep_mask &= ~cleaned.str.startswith(v_clean, na=False)
            elif mode == "endswith":
                keep_mask &= ~cleaned.str.endswith(v_clean, na=False)
        filtered = df[keep_mask]
        removed = len(df) - len(filtered)
        if removed > 0:
            # log locally for preview clarity
            try:
                self.preprocessor._log_transformation(f"{self.column}: locally removed {removed} rows matching '{raw}' (mode={mode})")
            except Exception:
                pass
        return filtered

    def _make_temp_preprocessor(self, df_for_preview:pd.DataFrame):
      
        if AdvancedPreprocessor is not None:
            try:
                tmp = AdvancedPreprocessor(df_for_preview)
                return tmp
            except Exception:
                pass
        # fallback: try shallow copy of provided preprocessor and swap df_original
        try:
            tmp = self.preprocessor.__class__(df_for_preview) if hasattr(self.preprocessor.__class__,"__call__") else None
        except Exception:
            tmp = None
        if tmp is None:
            class Wrapper:
                def __init__(self, source_df, original_pre):
                    self.df_original = source_df.copy()
                    self._original = original_pre
                def apply_advanced_transformations(self, column_settings, global_settings, selected_features=None, target=None):
                    # call original preprocessor but temporarily swap its df_original
                    orig = None
                    if hasattr(self._original, "df_original"):
                        orig = self._original.df_original
                        self._original.df_original = self.df_original
                    try:
                        return self._original.apply_advanced_transformations(column_settings, global_settings, selected_features, target)
                    finally:
                        if orig is not None:
                            self._original.df_original = orig
                def get_transformations_log(self):
                    try:
                        return self._original.get_transformations_log()
                    except Exception:
                        return []
                def get_column_stats(self, column):
                    try:
                        return self._original.get_column_stats(column)
                    except Exception:
                        s = self.df_original[column]
                        return {'dtype':str(s.dtype),'total_rows':len(s),'missing_values':int(s.isna().sum()),'unique_values':int(s.nunique())}
            return Wrapper(df_for_preview, self.preprocessor)
        return tmp

    def _on_setting_change(self, event=None):
        if self._updating:
            return
        self._updating = True
        try:
            settings = self._collect_settings()
            
            # Update the current settings
            self.current_settings.update(settings)
            
            # Notify parent callback to update main preview
            try:
                self.on_settings_change(self.column, self.current_settings)
            except Exception as e:
                print(f"Error in callback: {e}")
            
            # Optional: Keep local preview if you want it
            # self._update_local_preview()
        # ⭐ PART 4 — Update the right-side logs
            self._update_log()

            # ⭐ PART 2 — Update mapping tab logs (if mapping tab is visible)
            if hasattr(self, "_update_mapping_logs"):
                self._update_mapping_logs()
            
        finally:
            self.dialog.after(60, lambda: setattr(self, "_updating", False))

    def _update_local_preview(self):
        """Update the local preview in advanced editor if needed"""
        try:
            # Simple implementation - you can enhance this if you want local preview
            # For now, we'll rely on the main preprocessing preview
            pass
        except Exception as e:
            print(f"Local preview error: {e}")

    def _display_preview_error(self, err):
        self.processed_text = None
        self.log_text.config(state="normal"); self.log_text.delete("1.0","end"); self.log_text.insert("1.0", f"Preview error: {err}"); self.log_text.config(state="disabled")
        # clear treeviews
        try:
            for t in (self.tree_original, self.tree_processed):
                if t is not None:
                    for iid in t.get_children():
                        t.delete(iid)
        except Exception:
            pass

    def _populate_treeviews(self, orig_df:pd.DataFrame, proc_df:pd.DataFrame):
        # Helper to populate a treeview from a dataframe (show up to N rows & columns)
        def populate_tree(tree, df):
            # clear
            for iid in tree.get_children():
                tree.delete(iid)
            # set columns to df columns (string names)
            cols = list(df.columns)
            if not cols:
                tree["columns"] = ("val",)
                tree.heading("val", text="(no columns)")
                tree.column("val", width=200, stretch=True)
                return
            # Flatten columns to strings for treeview
            str_cols = [str(c) for c in cols]
            tree["columns"] = str_cols
            tree["show"] = "headings"
            # Configure headings and column widths
            for c in str_cols:
                tree.heading(c, text=c, anchor="w")
                # set a default width based on sample content
                maxlen = max([len(str(x)) for x in df[c].head(50).astype(str).tolist()] + [len(c), 10])
                width = min(max(80, maxlen*8), 600)
                tree.column(c, width=width, stretch=True, anchor="w")
            # insert rows (limit to max_rows to keep UI snappy)
            max_rows = 500 if len(df) > 500 else len(df)
            # if too many columns, show only head rows but still full columns
            rows_to_show = df.head(max_rows).itertuples(index=False, name=None)
            for r in rows_to_show:
                # convert nan to empty
                vals = [("" if (isinstance(v,float) and math.isnan(v)) else v) for v in r]
                tree.insert("", "end", values=vals)
            # enable column resizing by user; Treeview supports it by default on headings

        # Populate original and processed trees
        try:
            populate_tree(self.tree_original, orig_df)
        except Exception as e:
            # fallback: show minimal info
            try:
                self.tree_original["columns"] = ("val",); self.tree_original.heading("val", text="(error showing original)")
            except Exception:
                pass
        try:
            populate_tree(self.tree_processed, proc_df)
        except Exception as e:
            try:
                self.tree_processed["columns"] = ("val",); self.tree_processed.heading("val", text="(error showing processed)")
            except Exception:
                pass

    def _update_log(self):
        try:
            logs = self.preprocessor.get_transformations_log() or []
        except Exception:
            logs = []
        self.log_text.config(state="normal"); self.log_text.delete("1.0","end")
        for l in logs:
            self.log_text.insert("end", f"• {l}\n")
        self.log_text.config(state="disabled")

    def get_settings(self):
        return dict(self.current_settings)
    
    def _set_all_remove_vars(self, val: bool):
        """Set all remove checkboxes to the same value"""
        for v in self.remove_value_vars.values():
            v.set(val)
        self._on_remove_selection_change()

    def _on_remove_selection_change(self):
        """Update the remove values string when checkboxes change - IMPROVED VERSION"""
        selected_values = [v for v, var in self.remove_value_vars.items() if var.get()]
        
        # Update the remove_values_var
        self.remove_values_var.set(", ".join(selected_values))
        
        # Force immediate settings update and preview refresh
        self._on_setting_change()
        
    def _apply_all_settings(self):
        """Apply all current settings to parent immediately."""
        try:
            self.on_settings_change(self.column, self._collect_settings())
            messagebox.showinfo("Applied", "All settings applied successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}")

    def _clear_all_settings(self):
        """Reset ALL settings to default."""
        for key in self.current_settings.keys():
            if key in ("orientation",):
                continue
            self.current_settings[key] = ""

        # Refresh UI → rebuild
        self._create_ui()

        # Notify parent
        self.on_settings_change(self.column, self.current_settings)

        messagebox.showinfo("Cleared", "All transformations removed.")

    def _on_apply_transformations(self):
        """Apply transformations and update system settings"""
        try:
            # Get current settings
            settings = self._collect_settings()
            
            # Update the transformation log immediately
            transformation_summary = self._generate_transformation_summary(settings)
            
            # Update parent system with new settings
            if hasattr(self, 'on_settings_change'):
                self.on_settings_change(self.column, settings)
            
            # Show success message
            messagebox.showinfo("Applied", f"Transformations applied successfully!\n\n{transformation_summary}")
            
            # Refresh the UI to show updated state
            self._refresh_ui_with_new_settings(settings)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply transformations: {str(e)}")

    def _generate_transformation_summary(self, settings):
        """Generate a summary of applied transformations"""
        summary_parts = []
        
        if settings.get("dtype_conversion"):
            summary_parts.append(f"Type: → {settings['dtype_conversion']}")
        
        if settings.get("custom_mapping"):
            mapping_count = len([m for m in settings['custom_mapping'].split(',') if ':' in m])
            summary_parts.append(f"Value Mapping: {mapping_count} rules")
        
        if settings.get("range_mapping"):
            range_count = len([r for r in settings['range_mapping'].split(',') if ':' in r])
            summary_parts.append(f"Range Mapping: {range_count} ranges")
        
        if settings.get("strip_substring"):
            summary_parts.append(f"Strip: '{settings['strip_substring']}'")
        
        return "\n".join(summary_parts) if summary_parts else "No transformations applied"

    def _refresh_ui_with_new_settings(self, settings):
        """Refresh UI components to reflect new settings"""
        # Update the settings display in the main window
        if hasattr(self, 'per_col_widgets') and self.column in self.per_col_widgets:
            col_widgets = self.per_col_widgets[self.column]
            
            # Update summary label
            summary = self._get_settings_summary(self.column)
            if "summary_label" in col_widgets:
                col_widgets["summary_label"].config(text=summary)
        
        # Force refresh of the preview
        self._on_setting_change()

# If run directly, write a quick test UI
if __name__ == "__main__":
    import numpy as np
    df = pd.DataFrame({
        "type": np.random.choice(["Hybrid","Mono","Poly","Hybrid","Unknown","Hybrid "], size=300),
        "value": np.random.randn(300)*10 + 50,
        "desc": np.random.choice(["ok","bad","fine","needs cleaning"], size=300)
    })
    class DummyPre:
        def __init__(self, df): self.df_original = df.copy(); self._logs=[]
        def apply_advanced_transformations(self, cs, gs, selected_features=None, target=None):
            # Very simplified demo: just copy and apply mapping if any
            df = self.df_original.copy()
            s = cs.get("type", {})
            # custom mapping
            cm = s.get("custom_mapping","").strip()
            if cm:
                mapping = {}
                for item in cm.split(","):
                    if ":" in item:
                        k,v=item.split(":",1); mapping[k.strip()] = v.strip()
                if mapping:
                    df["type"] = df["type"].astype(str).replace(mapping)
                    self._logs.append(f"type: mapping applied ({len(mapping)} pairs)")
            return df
        def get_transformations_log(self): logs = list(self._logs); self._logs=[]; return logs
        def get_column_stats(self,column):
            s = self.df_original[column]; return {'dtype':str(s.dtype),'total_rows':len(s),'missing_values':int(s.isna().sum()), 'unique_values':int(s.nunique())}

    root = tk.Tk(); root.withdraw()
    p = DummyPre(df)
    ed = AdvancedEditor(root, p, "type", {}, lambda c,s: print("changed",c,s))
    root.mainloop()


