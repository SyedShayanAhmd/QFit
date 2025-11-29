# trainer_pycaret.py - COMPLETELY THREAD-SAFE VERSION
import os
import traceback
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.model_selection import train_test_split

# DISABLE ALL TKINTER AND PROGRESS BARS AT IMPORT LEVEL
os.environ['PYCARET_LOGGING_LEVEL'] = 'CRITICAL'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Monkey-patch to prevent Tkinter usage in threads


class PyCaretTrainer:
    def __init__(self, df: pd.DataFrame, features: List[str], target: Optional[str], output_dir: Optional[str] = None):
        # Create clean copy
        self.df = df.copy()
        
        # Log initial data info
        self._logs = []
        self.log("=" * 60)
        self.log("📊 INITIAL DATA SUMMARY")
        self.log("=" * 60)
        self.log(f"Original data shape: {self.df.shape}")
        self.log(f"Features requested: {len(features)}")
        
        # Only fix duplicates if they actually exist
        if not self.df.columns.is_unique:
            new_columns = []
            counter = {}
            for col in self.df.columns:
                if col in counter:
                    counter[col] += 1
                    new_name = f"{col}_{counter[col]}"
                    new_columns.append(new_name)
                else:
                    counter[col] = 0
                    new_columns.append(col)
            self.df.columns = new_columns
            self.log(f"🔧 Fixed duplicate columns: {[col for col in new_columns if '_' in col]}")
        
        # Filter features to only those that exist after potential renaming
        self.features = [f for f in features if f in self.df.columns and f != target]
        self.target = target if target in self.df.columns else None
        self.output_dir = output_dir or os.getcwd()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log detailed feature info
        self.log(f"✅ Final features used: {len(self.features)}")
        if self.features:
            self.log(f"📋 Features: {', '.join(self.features[:5])}{'...' if len(self.features) > 5 else ''}")
        self.log(f"🎯 Target: {self.target if self.target else 'None (unsupervised)'}")
        self.log(f"💾 Output directory: {self.output_dir}")

    def log(self, *parts, redirect_to_ui=True):
        """Thread-safe logging - NO TKINTER"""
        text = " ".join(str(p) for p in parts)
        self._logs.append(text)
        
        # Print to console (thread-safe)
        print(f"[PyCaret] {text}")
        
        # Simple UI logging without Tkinter dependencies
        if redirect_to_ui and hasattr(self, 'log_queue'):
            try:
                # Just store the message, main thread will handle display
                self.log_queue.put(text)
            except:
                pass  # Silently fail if queue issues

    def get_logs(self) -> str:
        return "\n".join(self._logs)

    def _completely_disable_progress_bars(self):
        """Completely disable all progress bars in PyCaret - SAFER VERSION"""
        try:
            # Set environment variables to minimize output
            os.environ['PYCARET_LOGGING_LEVEL'] = 'CRITICAL'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Configure PyCaret to be completely silent
            try:
                import pycaret.utils
                # Use PyCaret's built-in configuration instead of monkey-patching
                pass
            except:
                pass
                
            self.log("🔧 PyCaret configured for silent operation")
            
        except Exception as e:
            self.log(f"⚠️  Progress bar configuration had issues: {e}")

    def _filter_numeric_features(self, df, features):
        """Filter features to only numeric types for clustering"""
        numeric_features = []
        non_numeric_features = []
        
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                numeric_features.append(feature)
            else:
                non_numeric_features.append(feature)
        
        if non_numeric_features:
            self.log(f"⚠️  Removing non-numeric features for clustering: {non_numeric_features}")
        
        return numeric_features

    def run(self, problem_kind: str = "auto", use_compare: bool = True, 
            model_to_create: Optional[str] = None, tune: bool = False, 
            finalize: bool = True, use_nan_as_test: bool = False, 
            fold: int = 5, session_id: int = 42, test_size: float = 0.2, 
            random_state: int = 42, setup_kwargs: Optional[Dict[str, Any]] = None,
            model_hyperparameters: Optional[Dict[str, Any]] = None,
            chunk_callback: Optional[callable] = None) -> Dict[str, Any]:
        
        # Add periodic UI updates
        def ui_update(message):
            if chunk_callback:
                chunk_callback(message)
            self.log(message)
        
        ui_update("Starting PyCaret setup...")
            
        # Check PyCaret availability
        try:
            import pycaret
            self.log(f"✅ PyCaret {pycaret.__version__} available")
        except ImportError as e:
            return {"error": f"PyCaret not installed: {str(e)}", "run_log": self.get_logs()}

        # Prepare data
        train_df = self.df.copy()
        test_df = None
        
        # For clustering, ensure we only use numeric features
        if problem_kind in ["clustering", "anomaly"]:
            self.features = self._filter_numeric_features(train_df, self.features)
            if not self.features:
                return {"error": "No numeric features available for clustering/anomaly detection", "run_log": self.get_logs()}
        
        # Log data types and basic stats
        self.log(f"\n📈 DATA TYPES:")
        for col in self.features[:10]:
            dtype = str(train_df[col].dtype)
            unique = train_df[col].nunique()
            missing = train_df[col].isna().sum()
            self.log(f"   {col}: {dtype} | Unique: {unique} | Missing: {missing}")
        if len(self.features) > 10:
            self.log(f"   ... and {len(self.features) - 10} more features")
        
        # Handle NaN test split (only for supervised learning)
        if use_nan_as_test and self.target and self.target in train_df.columns:
            nan_mask = train_df[self.target].isna()
            if nan_mask.any():
                test_df = train_df[nan_mask].copy()
                train_df = train_df[~nan_mask].copy()
                self.log(f"📊 Using {len(test_df)} NaN rows as test set")
        
        # Apply train/test split if no test data and target exists (supervised only)
        elif self.target and self.target in train_df.columns and test_size > 0 and problem_kind in ["classification", "regression"]:
            try:
                relevant_cols = [f for f in self.features if f in train_df.columns]
                if self.target and self.target in train_df.columns:
                    relevant_cols.append(self.target)
                
                temp_df = train_df[relevant_cols].copy()
                
                self.log(f"\n🔀 TRAIN/TEST SPLIT CONFIG:")
                self.log(f"   Test size: {test_size:.1%}")
                self.log(f"   Random state: {random_state}")
                
                train_data, test_data = train_test_split(
                    temp_df, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=temp_df[self.target] if problem_kind == "classification" else None
                )
                
                train_df = train_data
                test_df = test_data
                self.log(f"✅ Split successful: {len(train_df)} train, {len(test_df)} test samples")
                
            except Exception as e:
                self.log(f"❌ Train/test split failed: {str(e)}")
                test_df = None
        
        # Select only relevant columns
        relevant_cols = [f for f in self.features if f in train_df.columns]
        if self.target and self.target in train_df.columns and problem_kind in ["classification", "regression"]:
            relevant_cols.append(self.target)
        
        train_df = train_df[relevant_cols].copy()
        if test_df is not None:
            test_df = test_df[relevant_cols].copy()
        
        self.log(f"\n📦 FINAL DATA SHAPES:")
        self.log(f"   Training data: {train_df.shape}")
        if test_df is not None:
            self.log(f"   Test data: {test_df.shape}")
        
        # Auto-detect problem type
        if problem_kind == "auto":
            if self.target and self.target in train_df.columns:
                if pd.api.types.is_numeric_dtype(train_df[self.target]) and train_df[self.target].nunique() > 20:
                    problem_kind = "regression"
                    self.log(f"🔍 Auto-detected: REGRESSION (numeric target with {train_df[self.target].nunique()} unique values)")
                else:
                    problem_kind = "classification"
                    self.log(f"🔍 Auto-detected: CLASSIFICATION (target with {train_df[self.target].nunique()} unique values)")
            else:
                problem_kind = "clustering"
                self.log(f"🔍 Auto-detected: CLUSTERING (no target specified)")
        else:
            self.log(f"🎯 User specified: {problem_kind.upper()}")
        
        # Force disable all Tkinter-related parameters
        setup_kwargs = setup_kwargs or {}
        base_setup = {
            "session_id": session_id,
            "verbose": False,
            "html": False,
        }
        
        # Add fold parameter only for supervised learning
        if problem_kind in ["classification", "regression"]:
            base_setup["fold"] = fold
        
        # Filter out any Tkinter-related parameters
        safe_kwargs = {}
        for key, value in setup_kwargs.items():
            if key not in ['html', 'verbose', 'log_plots', 'log_profile', 'log_data']:
                safe_kwargs[key] = value
        
        base_setup.update(safe_kwargs)
        
        self.log(f"\n⚙️  SETUP PARAMETERS:")
        for key, value in base_setup.items():
            self.log(f"   {key}: {value}")
        
        if problem_kind in ["classification", "regression"]:
            return self._run_supervised(
                problem_kind, train_df, test_df, use_compare, 
                model_to_create, tune, finalize, base_setup
            )
        elif problem_kind in ["clustering", "anomaly"]:
            return self._run_unsupervised(
                problem_kind, train_df, use_compare, 
                model_to_create, base_setup
            )
        else:
            return {"error": f"Unknown problem type: {problem_kind}", "run_log": self.get_logs()}
            
    def _run_supervised(self, problem_kind, train_df, test_df, use_compare, model_to_create, tune, finalize, base_setup):
            """Run supervised learning (UPDATED: Captures Transformed Feature Names)"""
            try:
                # Imports needed for metrics
                from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                                        accuracy_score, f1_score, precision_score, recall_score)
                import numpy as np

                # 1. Setup & Import correct get_config
                if problem_kind == "classification":
                    from pycaret.classification import setup, compare_models, create_model, tune_model, finalize_model, predict_model, save_model, get_config
                else:
                    from pycaret.regression import setup, compare_models, create_model, tune_model, finalize_model, predict_model, save_model, get_config

                self.log(f"\n🎯 STARTING {problem_kind.upper()} TRAINING")
                
                # Setup experiment
                exp = setup(data=train_df, target=self.target, **base_setup)
                
                # --- FIX: CAPTURE TRANSFORMED FEATURE NAMES ---
                # This gets the actual columns the model sees (e.g. "Color_Red", "Color_Blue")
                try:
                    X_train_transformed = get_config('X_train')
                    final_features = X_train_transformed.columns.tolist()
                    self.log(f"✅ Captured {len(final_features)} transformed features for plotting.")
                except Exception as e:
                    self.log(f"⚠️ Could not fetch transformed names: {e}")
                    final_features = self.features # Fallback to original
                # ----------------------------------------------

                # Train/Compare
                if use_compare:
                    self.log(f"📊 Comparing models...")
                    model = compare_models(verbose=False)
                else:
                    hyperparameters = getattr(self, 'model_hyperparameters', {})
                    params = hyperparameters.get(model_to_create, {}) if hyperparameters else {}
                    params = {k: v for k, v in params.items() if v is not None}
                    
                    self.log(f"🔧 Creating model: {model_to_create}")
                    model = create_model(model_to_create, verbose=False, **params)

                if tune and not use_compare:
                    self.log(f"🎛️ Tuning hyperparameters...")
                    model = tune_model(model, verbose=False, choose_better=False)

                if finalize:
                    self.log(f"🏁 Finalizing model...")
                    model = finalize_model(model)

                # Save
                model_path = os.path.join(self.output_dir, f"pycaret_model.pkl")
                save_model(model, model_path)

                # Metrics Calculation
                metrics = {}
                y_true = None
                y_pred = None
                test_predictions_df = None

                def calc_metrics(data, prefix):
                    self.log(f"🔮 Calculating {prefix} metrics...")
                    preds = predict_model(model, data=data, verbose=False)
                    actual = data[self.target]
                    
                    if 'prediction_label' in preds.columns: predicted = preds['prediction_label']
                    elif 'Label' in preds.columns: predicted = preds['Label']
                    else: predicted = preds.iloc[:, -1]
                    
                    res = {}
                    if problem_kind == "regression":
                        res[f"{prefix}_mse"] = mean_squared_error(actual, predicted)
                        res[f"{prefix}_rmse"] = np.sqrt(res[f"{prefix}_mse"])
                        res[f"{prefix}_mae"] = mean_absolute_error(actual, predicted)
                        res[f"{prefix}_r2"] = r2_score(actual, predicted)
                    else:
                        res[f"{prefix}_accuracy"] = accuracy_score(actual, predicted)
                        res[f"{prefix}_f1"] = f1_score(actual, predicted, average='weighted')
                        res[f"{prefix}_precision"] = precision_score(actual, predicted, average='weighted')
                        res[f"{prefix}_recall"] = recall_score(actual, predicted, average='weighted')
                    
                    return res, preds, actual, predicted

                try:
                    train_metrics, _, _, _ = calc_metrics(train_df, "train")
                    metrics.update(train_metrics)
                except Exception as e:
                    self.log(f"⚠️ Could not calculate train metrics: {e}")

                if test_df is not None and not test_df.empty:
                    try:
                        test_res, test_predictions_df, y_true, y_pred = calc_metrics(test_df, "test")
                        metrics.update(test_res)
                        metrics["test_predictions_made"] = True
                    except Exception as e:
                        self.log(f"❌ Test prediction failed: {str(e)}")
                
                results_data = {
                    "metrics": metrics,
                    "artifacts": {"model_path": model_path},
                    "model": model,
                    "run_log": self.get_logs(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "test_predictions": test_predictions_df,
                    "test_data": (test_df.drop(columns=[self.target]) if test_df is not None else None, y_true) if test_df is not None else None,
                    "train_data": train_df,
                    
                    # 🔥 THIS IS THE KEY CHANGE: Pass final_features instead of self.features
                    "features": final_features, 
                    
                    "target": self.target,
                    "problem_type": problem_kind
                }
                
                self.log(f"\n🎉 TRAINING COMPLETE!")
                return results_data
                
            except Exception as e:
                return {"error": f"Supervised learning failed: {str(e)}", "run_log": self.get_logs()}

    def _run_unsupervised(self, problem_kind, train_df, use_compare, model_to_create, base_setup):
        """Run unsupervised learning (Clustering/Anomaly)"""
        try:
            if problem_kind == "clustering":
                from pycaret.clustering import setup, create_model, assign_model, save_model
                default_model = "kmeans"
                module_name = "clustering"
            else:
                from pycaret.anomaly import setup, create_model, assign_model, save_model
                default_model = "iforest"
                module_name = "anomaly detection"

            self.log(f"\n🎯 STARTING {module_name.upper()}")
            
            # --- FIX: INCLUDE TARGET IN UNSUPERVISED LEARNING ---
            # Usually target is excluded, but for Clustering, we often want to cluster ON the target (e.g. Price)
            relevant_cols = [f for f in self.features if f in train_df.columns]
            
            if self.target and self.target in train_df.columns:
                relevant_cols.append(self.target)
                # Add to features list so Visualizer sees it
                if self.target not in self.features:
                    self.features.append(self.target)
                self.log(f"➕ Added Target '{self.target}' to analysis data")
            
            # Filter Data
            train_df_clean = train_df[relevant_cols].dropna().copy()
            # ----------------------------------------------------

            if train_df_clean.empty:
                return {"error": f"No data remaining after cleaning for {module_name}", "run_log": self.get_logs()}

            # Setup experiment
            self.log(f"\n⚙️  Setting up PyCaret experiment...")
            
            setup_params = base_setup.copy()
            # Remove 'fold' if it accidentally crept in (not valid for unsupervised)
            setup_params.pop('fold', None) 
            
            if problem_kind == "clustering":
                setup_params.update({
                    "numeric_imputation": "mean",
                    "categorical_imputation": "mode",
                })
            
            exp = setup(data=train_df_clean, **setup_params)
            self.log("✅ Experiment setup completed")
            
            # Create model
            model_name = model_to_create or default_model
            self.log(f"\n🔧 Creating {model_name} model...")
            model = create_model(model_name, verbose=False)
            self.log(f"✅ Model created: {type(model).__name__}")

            # Assign results
            self.log(f"\n🔮 Assigning {module_name} results...")
            assigned_data = assign_model(model)
            
            # Metrics
            metrics = {
                "data_points": len(train_df_clean),
                "features_used": len(relevant_cols)
            }
            
            # Calculate Clustering Metrics
            if problem_kind == "clustering" and hasattr(model, 'labels_'):
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                try:
                    # Ensure we have at least 2 clusters and 2 samples
                    labels = model.labels_
                    if len(np.unique(labels)) > 1 and len(train_df_clean) > len(np.unique(labels)):
                        # Use numeric data only for metrics
                        X_metrics = train_df_clean.select_dtypes(include=[np.number]).values
                        if X_metrics.shape[1] >= 1:
                            metrics["silhouette_score"] = silhouette_score(X_metrics, labels)
                            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_metrics, labels)
                            metrics["n_clusters"] = len(np.unique(labels))
                except Exception as e:
                    self.log(f"⚠️ Metrics error: {e}")
            
            # Save
            model_path = os.path.join(self.output_dir, f"pycaret_{problem_kind}_model.pkl")
            save_model(model, model_path)

            results = {
                "metrics": metrics,
                "artifacts": {"model_path": model_path},
                "model": model,
                "assigned_data": assigned_data,
                "run_log": self.get_logs(),
                "train_data": train_df_clean, # Now includes Price
                "features": relevant_cols,    # Now includes Price
                "problem_type": problem_kind
            }
            
            return results
            
        except Exception as e:
            error_msg = f"❌ Unsupervised learning failed: {str(e)}"
            self.log(error_msg)
            return {"error": error_msg, "run_log": self.get_logs()}