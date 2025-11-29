# ========= FILE: enhanced_preprocessor.py =========

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedPreprocessor:
    def __init__(self, parent, preprocessor, on_data_update, on_final_apply):
        self.parent = parent
        self.preprocessor = preprocessor
        self.on_data_update = on_data_update
        self.on_final_apply = on_final_apply
        
        # 🔥 CRITICAL FIX: Create a PROPER deep copy that persists
        self.df_current = preprocessor.df_original.copy(deep=True)  # Use deep copy
        self.pca_model = None
        self.pca_components = None
        self.original_columns = list(preprocessor.df_original.columns)
        
        # 🔥 Track data modifications
        self.data_modifications = []
        
        print(f"DEBUG: EnhancedPreprocessor initialized with shape: {self.df_current.shape}")
        print(f"DEBUG: Column dtypes: {self.df_current.dtypes.to_dict()}")
        
        self._create_ui()
        
    def _create_ui(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Enhanced Preprocessor - PCA & Python Editor")
        self.dialog.geometry("1200x800")
        self.dialog.transient(self.parent)
        
        # Main notebook
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 🔥 MONITOR tab changes to catch data reversion
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        # PCA Tab
        pca_tab = ttk.Frame(notebook)
        self._create_pca_tab(pca_tab)
        notebook.add(pca_tab, text="PCA Analysis")
        
        # Python Editor Tab
        python_tab = ttk.Frame(notebook)
        self._create_python_editor_tab(python_tab)
        notebook.add(python_tab, text="Python Script Editor")
        
        # Data Types Tab
        dtypes_tab = ttk.Frame(notebook)
        self._create_dtypes_tab(dtypes_tab)
        notebook.add(dtypes_tab, text="Data Types Manager")
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply Changes to Main App", 
                  command=self._apply_to_main_app, style="Accent.TButton").pack(side="right", padx=5)
        ttk.Button(button_frame, text="Update Preview", 
                  command=self._update_preview).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self._reset_data).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Close", 
                  command=self.dialog.destroy).pack(side="left", padx=5)
        
        # Status label
        self.status_label = ttk.Label(button_frame, text="Ready")
        self.status_label.pack(side="left", padx=10)
        
        self._refresh_data_info()

    def _on_tab_changed(self, event):
        """Handle tab changes to prevent data reversion"""
        print(f"🔄 Tab changed - verifying data integrity...")
        self._check_data_integrity("After tab change")
        
        # If we're switching to PCA tab, force refresh the list
        notebook = event.widget
        current_tab = notebook.select()
        tab_text = notebook.tab(current_tab, "text")
        
        if "PCA" in tab_text:
            print("🔄 Switching to PCA tab - refreshing feature list")
            self._force_refresh_pca_list_with_check()
        
    def _create_pca_tab(self, parent):
        # Main frame with scrollbar
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left: Controls
        left_frame = ttk.LabelFrame(main_frame, text="PCA Configuration")
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # Feature selection for PCA
        ttk.Label(left_frame, text="Select Features for PCA:").pack(anchor="w", pady=(10,5))
        
        # Frame for feature list with scrollbar
        feature_frame = ttk.Frame(left_frame)
        feature_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.feature_listbox = tk.Listbox(feature_frame, selectmode="multiple", height=10)
        feature_scrollbar = ttk.Scrollbar(feature_frame, orient="vertical", command=self.feature_listbox.yview)
        self.feature_listbox.configure(yscrollcommand=feature_scrollbar.set)
        
        self.feature_listbox.pack(side="left", fill="both", expand=True)
        feature_scrollbar.pack(side="right", fill="y")
        
        # Select all/none buttons
        select_frame = ttk.Frame(left_frame)
        select_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(select_frame, text="Select All", 
                  command=self._select_all_features).pack(side="left", padx=2)
        ttk.Button(select_frame, text="Select None", 
                  command=self._select_no_features).pack(side="left", padx=2)
        
        # Refresh button
        refresh_frame = ttk.Frame(left_frame)
        refresh_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(refresh_frame, text="🔄 Refresh Feature List", 
                command=self._force_refresh_pca_list).pack(fill="x", pady=2)
        
        ttk.Label(refresh_frame, text="Click refresh after data type changes", 
                font=("", 8), foreground="gray").pack()
        
        # Number of components
        ttk.Label(left_frame, text="Number of PCA Components:").pack(anchor="w", pady=(10,5))
        
        comp_frame = ttk.Frame(left_frame)
        comp_frame.pack(fill="x", padx=5, pady=5)
        
        self.n_components = tk.StringVar(value="3")
        components_combo = ttk.Combobox(comp_frame, textvariable=self.n_components,
                                      values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "custom"])
        components_combo.pack(side="left", fill="x", expand=True, padx=(0,5))
        components_combo.bind("<<ComboboxSelected>>", self._on_components_change)
        
        self.custom_components = tk.StringVar(value="")
        self.custom_entry = ttk.Entry(comp_frame, textvariable=self.custom_components, width=8)
        self.custom_entry.pack(side="left")
        self.custom_entry.pack_forget()  # Hide initially
        
        # Scaling option
        self.scale_data = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Scale data before PCA", 
                       variable=self.scale_data).pack(anchor="w", pady=5)
        
        # Buttons
        ttk.Button(left_frame, text="Run PCA Analysis", 
                  command=self._run_pca_analysis).pack(fill="x", pady=10)
        ttk.Button(left_frame, text="Apply PCA Transformation to Data", 
                  command=self._apply_pca).pack(fill="x", pady=5)
        
        # PCA Results
        results_frame = ttk.LabelFrame(left_frame, text="PCA Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=40)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.results_text.config(state="disabled")
        
        # Right: Variance plot and component info
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Variance plot frame
        plot_frame = ttk.LabelFrame(right_frame, text="Explained Variance")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.variance_text = scrolledtext.ScrolledText(plot_frame, height=10)
        self.variance_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.variance_text.config(state="disabled")

        # Component details frame
        comp_details_frame = ttk.LabelFrame(right_frame, text="Component Details")
        comp_details_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.components_text = scrolledtext.ScrolledText(comp_details_frame, height=10)
        self.components_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.components_text.config(state="disabled")
        
        # Populate feature list
        self._populate_feature_list()
        
    def _create_python_editor_tab(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Editor frame
        editor_frame = ttk.LabelFrame(main_frame, text="Python Script Editor")
        editor_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Info text
        info_text = """# Available variables:
# - df: Your dataframe (pd.DataFrame)
# - np: numpy
# - pd: pandas
# 
# Examples:
# df['new_col'] = df['col1'] + df['col2']
# df = df.dropna()
# df = df[df['age'] > 18]
# df = df.rename(columns={'old_name': 'new_name'})
"""
        
        self.editor_text = scrolledtext.ScrolledText(editor_frame, height=20)
        self.editor_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.editor_text.insert("1.0", info_text)
        
        # Button frame
        button_frame = ttk.Frame(editor_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Execute Script", 
                  command=self._execute_script).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Editor", 
                  command=self._clear_editor).pack(side="left", padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Execution Output")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.output_text.config(state="disabled")
        
    def _create_dtypes_tab(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Current dtypes
        current_frame = ttk.LabelFrame(main_frame, text="Current Data Types")
        current_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.dtypes_text = scrolledtext.ScrolledText(current_frame, height=15)
        self.dtypes_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.dtypes_text.config(state="disabled")
        
        # Conversion controls
        convert_frame = ttk.LabelFrame(main_frame, text="Data Type Conversion")
        convert_frame.pack(fill="x", padx=5, pady=5)
        
        # Column selection
        ttk.Label(convert_frame, text="Column:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.convert_column = ttk.Combobox(convert_frame, state="readonly")
        self.convert_column.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Target dtype
        ttk.Label(convert_frame, text="Convert to:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.target_dtype = ttk.Combobox(convert_frame, values=["int", "float", "str", "category", "bool"])
        self.target_dtype.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        self.target_dtype.set("float")
        
        # Convert button
        ttk.Button(convert_frame, text="Convert", 
                  command=self._convert_dtype).grid(row=0, column=4, padx=5, pady=5)
        
        convert_frame.columnconfigure(1, weight=1)
        convert_frame.columnconfigure(3, weight=1)
        
        # Force conversion button
        force_frame = ttk.Frame(main_frame)
        force_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(force_frame, text="Force Convert All Numeric Columns", 
                  command=self._force_convert_numeric).pack(side="left", padx=5)
        ttk.Button(force_frame, text="Detect and Fix Data Types", 
                  command=self._auto_fix_dtypes).pack(side="left", padx=5)
        
        # Update column list
        self._update_dtype_info()
            
    def _populate_feature_list(self):
        """Populate feature listbox with numeric columns - SUPER ROBUST VERSION"""
        self.feature_listbox.delete(0, tk.END)
        
        # Get numeric columns with comprehensive detection
        numeric_cols = []
        for col in self.df_current.columns:
            try:
                # Method 1: Check pandas dtype
                if pd.api.types.is_numeric_dtype(self.df_current[col]):
                    numeric_cols.append(col)
                    continue
                    
                # Method 2: Try to convert and check
                temp_series = pd.to_numeric(self.df_current[col], errors='coerce')
                if not temp_series.isna().all():
                    # If at least 50% conversion success, consider it numeric
                    conversion_rate = 1 - (temp_series.isna().sum() / len(self.df_current))
                    if conversion_rate >= 0.5:
                        numeric_cols.append(col)
                        continue
                        
            except Exception as e:
                print(f"DEBUG: Error checking column {col}: {e}")
                continue
        
        # Sort columns for better UX
        numeric_cols.sort()
        
        print(f"DEBUG: Found {len(numeric_cols)} numeric columns: {numeric_cols}")
        
        # Clear and repopulate the stored column names
        self.feature_column_names = []
        
        for col in numeric_cols:
            dtype = str(self.df_current[col].dtype)
            non_null = self.df_current[col].notna().sum()
            total = len(self.df_current[col])
            
            # Create display text
            display_text = f"{col} ({dtype}, {non_null}/{total} valid)"
            self.feature_listbox.insert(tk.END, display_text)
            self.feature_column_names.append(col)
            
        # Select all by default if we have features
        if numeric_cols:
            self._select_all_features()
            print(f"DEBUG: Selected all {len(numeric_cols)} numeric columns")
        else:
            print("DEBUG: No numeric columns found for PCA")
            
    def _refresh_pca_feature_list(self):
        """Refresh the PCA feature list - call this after any dtype changes"""
        print("DEBUG: Refreshing PCA feature list...")
        
        # Store current selections to restore them
        current_selections = self._get_selected_features()
        print(f"DEBUG: Current selections: {current_selections}")
        
        # Repopulate the feature list
        self._populate_feature_list()
        
        # Try to restore previous selections
        restored_count = 0
        if current_selections:
            for i in range(self.feature_listbox.size()):
                if i < len(self.feature_column_names):
                    feature_name = self.feature_column_names[i]
                    if feature_name in current_selections:
                        self.feature_listbox.selection_set(i)
                        restored_count += 1
                        
        print(f"DEBUG: Restored {restored_count} previous selections")
        
        # Update status
        numeric_count = len(self.feature_column_names)
        self._show_message(f"PCA list updated: {numeric_count} numeric features available", "info")
       
    def _select_all_features(self):
        """Select all features in the listbox"""
        self.feature_listbox.selection_clear(0, tk.END)
        for i in range(self.feature_listbox.size()):
            self.feature_listbox.selection_set(i)
            
    def _select_no_features(self):
        """Clear all selections"""
        self.feature_listbox.selection_clear(0, tk.END)
        
    def _on_components_change(self, event):
        """Show/hide custom components entry"""
        if self.n_components.get() == "custom":
            self.custom_entry.pack(side="left")
        else:
            self.custom_entry.pack_forget()
                
    def _run_pca_analysis(self):
        """Run PCA analysis and show results - ENHANCED VERSION"""
        selected_features = self._get_selected_features()
        
        if not selected_features:
            self._show_message("Please select at least one feature for PCA", "error")
            return
            
        # Double-check that selected features are actually numeric
        non_numeric_features = []
        valid_features = []
        
        for feature in selected_features:
            if feature in self.df_current.columns:
                if pd.api.types.is_numeric_dtype(self.df_current[feature]):
                    valid_features.append(feature)
                else:
                    non_numeric_features.append(feature)
        
        if non_numeric_features:
            self._show_message(f"These features are not numeric and will be excluded: {non_numeric_features}", "warning")
        
        if not valid_features:
            self._show_message("No valid numeric features selected for PCA", "error")
            return
            
        try:
            # Get number of components
            if self.n_components.get() == "custom":
                n_comp = int(self.custom_components.get())
            else:
                n_comp = int(self.n_components.get())
                
            if n_comp > len(valid_features):
                n_comp = len(valid_features)
                self._show_message(f"Reduced components to {n_comp} (number of valid features)", "info")
                
            # Prepare data
            X = self.df_current[valid_features].dropna()
            
            if len(X) == 0:
                self._show_message("No data available after removing NaN values", "error")
                return
                
            if len(X) < n_comp:
                self._show_message(f"Not enough samples ({len(X)}) for {n_comp} components", "error")
                return
                
            if self.scale_data.get():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
                
            # Run PCA
            self.pca_model = PCA(n_components=n_comp)
            principal_components = self.pca_model.fit_transform(X_scaled)
            self.pca_components = principal_components
            
            # Update results
            self._update_pca_results(valid_features)
            self._show_message(f"PCA analysis completed! Used {len(valid_features)} numeric features", "success")
            
        except Exception as e:
            self._show_message(f"PCA Error: {str(e)}", "error")
            
    def _update_pca_results(self, features):
        """Update PCA results display"""
        # Results text
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        
        self.results_text.insert(tk.END, f"PCA Analysis Results\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        self.results_text.insert(tk.END, f"Features used: {len(features)}\n")
        self.results_text.insert(tk.END, f"Components: {self.pca_model.n_components_}\n")
        self.results_text.insert(tk.END, f"Explained variance ratio: {self.pca_model.explained_variance_ratio_.sum():.4f}\n")
        self.results_text.insert(tk.END, f"Singular values: {self.pca_model.singular_values_}\n")
        
        self.results_text.config(state="disabled")
        
        # Variance text
        self.variance_text.config(state="normal")
        self.variance_text.delete("1.0", tk.END)
        
        self.variance_text.insert(tk.END, "Explained Variance by Component:\n")
        self.variance_text.insert(tk.END, "=" * 40 + "\n")
        
        for i, variance in enumerate(self.pca_model.explained_variance_ratio_):
            self.variance_text.insert(tk.END, f"PC{i+1}: {variance:.4f} ({variance*100:.2f}%)\n")
            
        self.variance_text.insert(tk.END, f"\nCumulative Variance:\n")
        self.variance_text.insert(tk.END, "=" * 40 + "\n")
        
        cumulative = 0
        for i, variance in enumerate(self.pca_model.explained_variance_ratio_):
            cumulative += variance
            self.variance_text.insert(tk.END, f"PC{i+1}: {cumulative:.4f} ({cumulative*100:.2f}%)\n")
            
        self.variance_text.config(state="disabled")
        
        # Components text
        self.components_text.config(state="normal")
        self.components_text.delete("1.0", tk.END)
        
        self.components_text.insert(tk.END, "Component Loadings:\n")
        self.components_text.insert(tk.END, "=" * 40 + "\n")
        
        features = self._get_selected_features()
        for i in range(self.pca_model.n_components_):
            self.components_text.insert(tk.END, f"\nPC{i+1}:\n")
            loadings = self.pca_model.components_[i]
            
            # Sort by absolute value
            feature_loadings = list(zip(features, loadings))
            feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, loading in feature_loadings[:5]:  # Top 5 features
                self.components_text.insert(tk.END, f"  {feature}: {loading:.4f}\n")
                
        self.components_text.config(state="disabled")
        
    def _apply_pca(self):
        """Apply PCA transformation to data"""
        if self.pca_components is None:
            self._show_message("Please run PCA analysis first", "error")
            return
            
        try:
            # Create PCA column names
            n_components = self.pca_components.shape[1]
            pca_columns = [f"PC{i+1}" for i in range(n_components)]
            
            # Get the indices of non-NaN rows used for PCA
            selected_features = self._get_selected_features()
            valid_indices = self.df_current[selected_features].dropna().index
            
            # Create full PCA dataframe with NaN for rows that were excluded
            pca_df_full = pd.DataFrame(index=self.df_current.index, columns=pca_columns)
            pca_df_full.loc[valid_indices, pca_columns] = self.pca_components
            
            # Remove original features used for PCA
            df_without_pca_features = self.df_current.drop(columns=selected_features)
            
            # Combine with PCA components
            self.df_current = pd.concat([df_without_pca_features, pca_df_full], axis=1)
            
            self._show_message(f"PCA applied! Added {n_components} principal components", "success")
            self._update_preview()
            self._update_dtype_info()
            
        except Exception as e:
            self._show_message(f"Error applying PCA: {str(e)}", "error")
                
    def _execute_script(self):
        """Execute Python script on data"""
        script = self.editor_text.get("1.0", tk.END)
        
        try:
            # Create safe execution environment
            local_vars = {
                'df': self.df_current.copy(),
                'np': np,
                'pd': pd
            }
            
            # Execute script
            exec(script, globals(), local_vars)
            
            # Get modified dataframe
            if 'df' in local_vars:
                new_df = local_vars['df']
                
                # Validate result
                if not isinstance(new_df, pd.DataFrame):
                    raise ValueError("Script must result in a pandas DataFrame")
                    
                self.df_current = new_df
                self._show_message("Script executed successfully!", "success")
                self._update_preview()
                self._update_dtype_info()
                
                # 🔥 FORCE PCA LIST REFRESH
                self._force_refresh_pca_list()
                
                # Update output text
                self.output_text.config(state="normal")
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert(tk.END, "Script executed successfully!\n")
                self.output_text.insert(tk.END, f"New shape: {new_df.shape}\n")
                self.output_text.insert(tk.END, f"New columns: {list(new_df.columns)}")
                self.output_text.config(state="disabled")
                
            else:
                self._show_message("Warning: No 'df' variable found after execution", "warning")
                
        except Exception as e:
            error_msg = f"Script Error: {str(e)}"
            self._show_message(error_msg, "error")
            
            # Show error in output
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, error_msg)
            self.output_text.config(state="disabled")
            
    def _clear_editor(self):
        """Clear the script editor"""
        self.editor_text.delete("1.0", tk.END)
        
    def _convert_dtype(self):
        """Convert data type of selected column - WITH PERSISTENCE FIX"""
        column = self.convert_column.get()
        target_type = self.target_dtype.get()
        
        if not column:
            self._show_message("Please select a column", "error")
            return
            
        try:
            # 🔥 CRITICAL: Check current state
            original_dtype = str(self.df_current[column].dtype)
            print(f"DEBUG: Converting {column} from {original_dtype} to {target_type}")
            
            # Make a copy to ensure we're working with the right data
            series_copy = self.df_current[column].copy()
            
            if target_type == "int":
                converted = pd.to_numeric(series_copy, errors='coerce')
                # Check if conversion was successful
                if converted.isna().all():
                    self._show_message(f"Failed to convert {column} to int - all values became NaN", "error")
                    return
                self.df_current[column] = converted.astype('Int64')
                
            elif target_type == "float":
                converted = pd.to_numeric(series_copy, errors='coerce')
                if converted.isna().all():
                    self._show_message(f"Failed to convert {column} to float - all values became NaN", "error")
                    return
                self.df_current[column] = converted.astype(float)
                
            elif target_type == "str":
                self.df_current[column] = series_copy.astype(str)
            elif target_type == "category":
                self.df_current[column] = series_copy.astype('category')
            elif target_type == "bool":
                self.df_current[column] = series_copy.astype(bool)
            
            # 🔥 VERIFY the conversion actually happened
            new_dtype = str(self.df_current[column].dtype)
            print(f"DEBUG: Conversion result - {column}: {original_dtype} → {new_dtype}")
            
            if original_dtype == new_dtype and target_type not in ["str", "category", "bool"]:
                self._show_message(f"Warning: Conversion may have failed - dtype still {new_dtype}", "warning")
            
            self._show_message(f"Converted {column} from {original_dtype} to {new_dtype}", "success")
            
            # 🔥 CRITICAL: Run integrity check
            self._check_data_integrity(f"After converting {column} to {target_type}")
            
            # Update displays
            self._update_preview()
            self._update_dtype_info()
            
            # 🔥 FORCE PCA LIST REFRESH with verification
            self._force_refresh_pca_list_with_check()
            
            # Log the modification
            self.data_modifications.append(f"Converted {column} to {target_type}")
            
        except Exception as e:
            self._show_message(f"Conversion Error: {str(e)}", "error")
            print(f"ERROR in _convert_dtype: {e}")

    def _force_convert_numeric(self):
        """Force convert all possible columns to numeric"""
        converted = []
        for col in self.df_current.columns:
            if self.df_current[col].dtype == 'object':
                # Try to convert to numeric
                original_dtype = str(self.df_current[col].dtype)
                numeric_series = pd.to_numeric(self.df_current[col], errors='coerce')
                if not numeric_series.isna().all():
                    conversion_rate = 1 - (numeric_series.isna().sum() / len(self.df_current))
                    if conversion_rate >= 0.8:  # At least 80% success
                        self.df_current[col] = numeric_series
                        new_dtype = str(self.df_current[col].dtype)
                        converted.append(f"{col}: {original_dtype}→{new_dtype}")
                        
        if converted:
            self._show_message(f"Converted to numeric:\n" + "\n".join(converted), "success")
        else:
            self._show_message("No columns could be converted to numeric", "info")
            
        self._update_preview()
        self._update_dtype_info()
        
        # 🔥 FORCE PCA LIST REFRESH
        self._force_refresh_pca_list()

    def _auto_fix_dtypes(self):
        """Automatically detect and fix data types"""
        changes = []
        
        for col in self.df_current.columns:
            original_dtype = str(self.df_current[col].dtype)
            
            # Try numeric conversion for object columns
            if self.df_current[col].dtype == 'object':
                numeric_series = pd.to_numeric(self.df_current[col], errors='coerce')
                if not numeric_series.isna().all():
                    conversion_rate = 1 - (numeric_series.isna().sum() / len(self.df_current))
                    if conversion_rate >= 0.9:  # High success rate
                        self.df_current[col] = numeric_series
                        new_dtype = str(self.df_current[col].dtype)
                        changes.append(f"{col}: {original_dtype} → {new_dtype}")
                        
        if changes:
            self._show_message("Auto-fix applied:\n" + "\n".join(changes), "success")
        else:
            self._show_message("No data type changes needed", "info")
            
        self._update_preview()
        self._update_dtype_info()
        
        # 🔥 FORCE PCA LIST REFRESH
        self._force_refresh_pca_list()
        
    def _get_selected_features(self):
        """Get list of selected features from listbox - UPDATED VERSION"""
        selected_indices = self.feature_listbox.curselection()
        
        # Use the stored column names instead of listbox text
        selected_features = []
        for i in selected_indices:
            if i < len(self.feature_column_names):
                selected_features.append(self.feature_column_names[i])
        
        return selected_features
        
    def _update_preview(self):
        """Update the preview in main app"""
        if self.on_data_update:
            self.on_data_update(self.df_current)
        self._update_dtype_info()
        
    def _verify_data_consistency(self):
        """Emergency method to verify data hasn't reverted"""
        print("\n🚨 EMERGENCY DATA CONSISTENCY CHECK")
        
        # Check if any modifications have been lost
        if hasattr(self, 'data_modifications') and self.data_modifications:
            print(f"Tracked modifications: {self.data_modifications}")
        
        # Force refresh all displays
        self._check_data_integrity("Emergency check")
        self._update_dtype_info()
        self._force_refresh_pca_list_with_check()

    def _update_dtype_info(self):
        """Update data type information display - WITH VERIFICATION"""
        self.dtypes_text.config(state="normal")
        self.dtypes_text.delete("1.0", tk.END)
        
        self.dtypes_text.insert(tk.END, "Current Data Types (REAL-TIME):\n")
        self.dtypes_text.insert(tk.END, "=" * 70 + "\n\n")
        
        for col in self.df_current.columns:
            dtype = str(self.df_current[col].dtype)
            non_null = self.df_current[col].notna().sum()
            total = len(self.df_current[col])
            null_count = total - non_null
            null_percentage = (null_count / total) * 100 if total > 0 else 0
            
            # Show detailed info with verification
            self.dtypes_text.insert(tk.END, f"{col:<25} {dtype:<15} {non_null}/{total} non-null\n")
            self.dtypes_text.insert(tk.END, f"{'':<25} {'':<15} {null_count} missing ({null_percentage:.1f}%)\n")
            
            # Add verification status
            is_numeric = pd.api.types.is_numeric_dtype(self.df_current[col])
            status = "✅ NUMERIC" if is_numeric else "❌ OBJECT"
            self.dtypes_text.insert(tk.END, f"{'':<25} {'':<15} {status}\n")
            
            self.dtypes_text.insert(tk.END, "\n")
                
        self.dtypes_text.config(state="disabled")
        
        # Update conversion column combobox
        self.convert_column['values'] = list(self.df_current.columns)
        if len(self.df_current.columns) > 0:
            self.convert_column.set(self.df_current.columns[0])
            
    def _show_message(self, message, msg_type="info"):
        """Show message in status label"""
        self.status_label.config(text=message)
        
        # Color coding
        if msg_type == "error":
            self.status_label.config(foreground="red")
        elif msg_type == "success":
            self.status_label.config(foreground="green")
        elif msg_type == "warning":
            self.status_label.config(foreground="orange")
        else:
            self.status_label.config(foreground="black")
            
        # Also show in messagebox for important messages
        if msg_type == "error":
            messagebox.showerror("Error", message)
        elif msg_type == "success":
            messagebox.showinfo("Success", message)
            
    def _apply_to_main_app(self):
        """Apply all changes to main application"""
        try:
            # Update ALL dataframes in the main app
            if self.on_final_apply:
                success = self.on_final_apply(self.df_current)
                if success:
                    self._show_message("Changes applied to main application successfully!", "success")
                else:
                    self._show_message("Failed to apply changes to main application", "error")
            else:
                self._show_message("No callback available to apply changes", "error")
                
        except Exception as e:
            self._show_message(f"Error applying changes: {str(e)}", "error")
            
    def _refresh_data_info(self):
        """Refresh data information display"""
        self._update_preview()
        
    def _reset_data(self):
        """Reset data to original"""
        self.df_current = self.preprocessor.df_original.copy()
        self._show_message("Data reset to original", "info")
        self._update_preview()
        self._update_dtype_info()
        
    def _force_refresh_pca_list(self):
        """Force refresh PCA list and ensure it's visible"""
        print(f"DEBUG: Refreshing PCA list. Current columns: {list(self.df_current.columns)}")
        print(f"DEBUG: Numeric columns: {self.df_current.select_dtypes(include=[np.number]).columns.tolist()}")
        
        # Store current tab and switch to PCA tab to ensure list is visible
        current_tab = self.dialog.nametowidget(self.dialog.focus_get())
        
        # Force refresh the feature list
        self._refresh_pca_feature_list()
        
        # Update status
        numeric_count = len(self.df_current.select_dtypes(include=[np.number]).columns)
        self._show_message(f"PCA list refreshed. {numeric_count} numeric columns available", "info")

    def _is_truly_numeric(self, series):
        """Check if a series is truly numeric (more robust detection)"""
        try:
            # Already numeric dtype
            if pd.api.types.is_numeric_dtype(series):
                return True
                
            # Try conversion
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.isna().all():
                return False
                
            # Check if we have reasonable numeric values (not all 0, 1 like categorical)
            unique_values = numeric_series.dropna().unique()
            if len(unique_values) <= 2:
                # Could be binary categorical, check if values make sense as numeric
                if set(unique_values).issubset({0, 1}):
                    return True  # Accept binary as numeric for PCA
                else:
                    # Check if the values have meaningful numeric range
                    value_range = numeric_series.max() - numeric_series.min()
                    if value_range > 10:  # Reasonable numeric range
                        return True
                    else:
                        return False
            else:
                # Multiple unique values, likely numeric
                return True
                
        except:
            return False
        
    def _check_data_integrity(self, operation=""):
        """Check and log data integrity after operations"""
        print(f"\n🔍 DATA INTEGRITY CHECK - {operation}")
        print(f"Shape: {self.df_current.shape}")
        print(f"Columns: {list(self.df_current.columns)}")
        print("Dtypes:")
        for col in self.df_current.columns:
            dtype = str(self.df_current[col].dtype)
            non_null = self.df_current[col].notna().sum()
            print(f"  {col}: {dtype} ({non_null} non-null)")
        
        # Check for object columns that should be numeric
        object_cols = self.df_current.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"⚠️  Object columns found: {list(object_cols)}")
            
        return True
        
    def _force_refresh_pca_list_with_check(self):
        """Refresh PCA list with data integrity verification"""
        print("\n🔄 FORCE REFRESHING PCA LIST WITH VERIFICATION")
        
        # First, verify current data state
        self._check_data_integrity("Before PCA list refresh")
        
        # Store current selections
        current_selections = self._get_selected_features()
        
        # Clear and repopulate
        self.feature_listbox.delete(0, tk.END)
        
        # Get numeric columns with verification
        numeric_cols = []
        for col in self.df_current.columns:
            current_dtype = str(self.df_current[col].dtype)
            
            # Check if numeric
            is_numeric = pd.api.types.is_numeric_dtype(self.df_current[col])
            
            if is_numeric:
                numeric_cols.append(col)
                print(f"✅ PCA List: Adding {col} as numeric ({current_dtype})")
            else:
                print(f"❌ PCA List: Skipping {col} - not numeric ({current_dtype})")
        
        # Sort and populate
        numeric_cols.sort()
        self.feature_column_names = numeric_cols.copy()
        
        for i, col in enumerate(numeric_cols):
            dtype = str(self.df_current[col].dtype)
            non_null = self.df_current[col].notna().sum()
            display_text = f"{col} ({dtype}, {non_null} valid)"
            self.feature_listbox.insert(tk.END, display_text)
        
        # Restore selections
        restored_count = 0
        if current_selections:
            for i, col in enumerate(self.feature_column_names):
                if col in current_selections:
                    self.feature_listbox.selection_set(i)
                    restored_count += 1
                    
        print(f"🔄 PCA List: Loaded {len(numeric_cols)} numeric features, restored {restored_count} selections")
        
        # Update status
        self._show_message(f"PCA: {len(numeric_cols)} numeric features available", "info")