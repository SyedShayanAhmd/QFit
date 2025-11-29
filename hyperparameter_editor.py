# hyperparameter_editor.py
import tkinter as tk
from tkinter import ttk
import json

class HyperparameterEditor:
    def __init__(self, parent, model_name, current_params=None, on_parameters_save=None):
        self.parent = parent
        self.model_name = model_name
        self.on_parameters_save = on_parameters_save
        self.current_params = current_params or {}
        
        self.model_parameters = {
            "lr": {
                "fit_intercept": {"type": "boolean", "default": True, "help": "Whether to calculate the intercept"},
                "normalize": {"type": "boolean", "default": False, "help": "Whether to normalize the features"},
                "copy_X": {"type": "boolean", "default": True, "help": "Whether to copy X"},
            },
            "lasso": {
                "alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "help": "Constant that multiplies the L1 term"},
                "fit_intercept": {"type": "boolean", "default": True, "help": "Whether to calculate the intercept"},
                "normalize": {"type": "boolean", "default": False, "help": "Whether to normalize the features"},
            },
            "ridge": {
                "alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "help": "Regularization strength"},
                "fit_intercept": {"type": "boolean", "default": True, "help": "Whether to calculate the intercept"},
                "solver": {"type": "choice", "default": "auto", "choices": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], "help": "Solver to use"},
            },
            "rf": {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 1000, "help": "Number of trees in the forest"},
                "max_depth": {"type": "int", "default": None, "min": 1, "max": 100, "help": "Maximum depth of the tree"},
                "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "help": "Minimum number of samples required to split"},
                "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 20, "help": "Minimum number of samples required at leaf node"},
            },
            "xgboost": {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 1000, "help": "Number of gradient boosted trees"},
                "max_depth": {"type": "int", "default": 6, "min": 1, "max": 20, "help": "Maximum tree depth"},
                "learning_rate": {"type": "float", "default": 0.3, "min": 0.01, "max": 1.0, "help": "Boosting learning rate"},
                "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "help": "Subsample ratio of training instances"},
            },
            "lightgbm": {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 1000, "help": "Number of boosted trees"},
                "max_depth": {"type": "int", "default": -1, "min": -1, "max": 100, "help": "Maximum tree depth (-1 for no limit)"},
                "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "help": "Boosting learning rate"},
                "num_leaves": {"type": "int", "default": 31, "min": 2, "max": 256, "help": "Maximum number of leaves in one tree"},
            }
        }
        
        self._create_ui()
        
    def _create_ui(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(f"Hyperparameters - {self.model_name}")
        self.dialog.geometry("600x500")
        self.dialog.transient(self.parent)
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create scrollable canvas
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Parameters for this model
        params = self.model_parameters.get(self.model_name, {})
        self.param_vars = {}
        
        if not params:
            ttk.Label(self.scrollable_frame, text=f"No advanced parameters for {self.model_name}").pack(pady=20)
        else:
            for param_name, param_config in params.items():
                self._create_parameter_widget(param_name, param_config)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self._apply_parameters).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self._reset_defaults).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side="right", padx=5)
        
    def _create_parameter_widget(self, param_name, config):
        frame = ttk.LabelFrame(self.scrollable_frame, text=param_name)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Help text
        ttk.Label(frame, text=config.get("help", ""), font=("", 8), foreground="gray").pack(anchor="w", padx=5, pady=2)
        
        # Value input based on type
        param_type = config["type"]
        default_value = self.current_params.get(param_name, config.get("default"))
        
        if param_type == "boolean":
            var = tk.BooleanVar(value=default_value)
            ttk.Checkbutton(frame, variable=var, text="Enabled").pack(anchor="w", padx=5, pady=2)
            
        elif param_type == "int":
            var = tk.StringVar(value=str(default_value) if default_value is not None else "")
            subframe = ttk.Frame(frame)
            subframe.pack(fill="x", padx=5, pady=2)
            ttk.Label(subframe, text="Value:").pack(side="left")
            ttk.Entry(subframe, textvariable=var, width=10).pack(side="left", padx=5)
            if "min" in config and "max" in config:
                ttk.Label(subframe, text=f"Range: {config['min']} - {config['max']}", font=("", 8)).pack(side="left", padx=10)
                
        elif param_type == "float":
            var = tk.StringVar(value=str(default_value) if default_value is not None else "")
            subframe = ttk.Frame(frame)
            subframe.pack(fill="x", padx=5, pady=2)
            ttk.Label(subframe, text="Value:").pack(side="left")
            ttk.Entry(subframe, textvariable=var, width=10).pack(side="left", padx=5)
            if "min" in config and "max" in config:
                ttk.Label(subframe, text=f"Range: {config['min']} - {config['max']}", font=("", 8)).pack(side="left", padx=10)
                
        elif param_type == "choice":
            var = tk.StringVar(value=default_value)
            ttk.Combobox(frame, textvariable=var, values=config["choices"], state="readonly").pack(anchor="w", padx=5, pady=2)
        
        self.param_vars[param_name] = (var, config)
        
    def _apply_parameters(self):
        parameters = {}
        for param_name, (var, config) in self.param_vars.items():
            try:
                if config["type"] == "boolean":
                    parameters[param_name] = var.get()
                elif config["type"] == "int":
                    value = var.get()
                    parameters[param_name] = int(value) if value else None
                elif config["type"] == "float":
                    value = var.get()
                    parameters[param_name] = float(value) if value else None
                else:
                    parameters[param_name] = var.get()
            except (ValueError, TypeError):
                # Keep default if invalid
                parameters[param_name] = config.get("default")
        
        if self.on_parameters_save:
            self.on_parameters_save(parameters)
        
        self.dialog.destroy()
        
    def _reset_defaults(self):
        for param_name, (var, config) in self.param_vars.items():
            default = config.get("default")
            if config["type"] == "boolean":
                var.set(default)
            else:
                var.set(str(default) if default is not None else "")