import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (confusion_matrix, classification_report, 
                           mean_squared_error, r2_score, roc_curve, auc,
                           precision_recall_curve, silhouette_score)
from sklearn.tree import plot_tree, export_text
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import scipy.stats as stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class EnhancedVisualizer:
    def __init__(self, results, problem_type, model, test_data=None, train_data=None, features=None, target=None):
        self.results = results
        self.problem_type = problem_type
        self.model = model
        self.test_data = test_data
        self.train_data = train_data
        self.features = features or []
        self.target = target

    def _unwrap_model(self):
        model = self.model
        if isinstance(model, Pipeline):
            if 'actual_estimator' in model.named_steps:
                return model.named_steps['actual_estimator']
            elif hasattr(model, 'steps'):
                return model.steps[-1][1]
        return model

    def _has_test_predictions(self):
        return (self.test_data is not None and 
                self.results is not None and 
                'y_true' in self.results and 
                'y_pred' in self.results)

    def _get_test_predictions(self):
        return self.results['y_true'], self.results['y_pred']

    def create_comprehensive_plots(self, output_dir):
        plots = {}
        try:
            p = self._plot_feature_importance(output_dir)
            if p: plots['feature_importance'] = p
            
            if self.train_data is not None and self.problem_type != "clustering":
                p = self._plot_learning_curve(output_dir)
                if p: plots['learning_curve'] = p

            if self.problem_type == "classification":
                plots.update(self._create_classification_plots(output_dir))
            elif self.problem_type == "regression":
                plots.update(self._create_regression_plots(output_dir))
            elif self.problem_type == "clustering":
                plots.update(self._create_clustering_plots(output_dir))
                
        except Exception as e:
            print(f"Error in plot generation: {e}")
        return plots

    def _create_classification_plots(self, output_dir):
        plots = {}
        if self._has_test_predictions():
            p = self._plot_confusion_matrix(output_dir); 
            if p: plots['confusion_matrix'] = p
            if hasattr(self.model, 'predict_proba'):
                p = self._plot_roc_curve(output_dir); 
                if p: plots['roc_curve'] = p
                p = self._plot_precision_recall_curve(output_dir); 
                if p: plots['precision_recall_curve'] = p

        inner = self._unwrap_model()
        if hasattr(inner, 'tree_') or hasattr(inner, 'estimators_'):
            class_names = None
            if hasattr(inner, 'classes_'): class_names = [str(c) for c in inner.classes_]
            p = self.plot_complete_decision_tree(output_dir, max_depth=5, class_names=class_names)
            if p: plots['decision_tree'] = p
        return plots

    def _create_regression_plots(self, output_dir):
        plots = {}
        if self._has_test_predictions():
            p = self._plot_actual_vs_predicted_45(output_dir); 
            if p: plots['actual_vs_predicted'] = p
            p = self._plot_residual_analysis(output_dir); 
            if p: plots['residual_analysis'] = p
        p = self._plot_coefficients_equation(output_dir)
        if p: plots['model_equation'] = p
        return plots

    def _create_clustering_plots(self, output_dir):
        plots = {}
        inner = self._unwrap_model()
        if hasattr(inner, 'labels_') and self.train_data is not None:
            p = self._plot_cluster_diagram_2d(output_dir)
            if p: plots['cluster_diagram'] = p
        return plots
    
    # --- NEW: PCA TEXT EXPLANATION ---
    def get_pca_explanation_text(self):
        """Generates the mathematical equation of the PCA components."""
        try:
            if self.train_data is None: return "No data available for PCA analysis."
            
            X = self.train_data
            # Ensure numeric
            if isinstance(X, pd.DataFrame):
                X_num = X.select_dtypes(include=[np.number])
                cols = X_num.columns.tolist()
                X = X_num.values
            else:
                cols = [f"Feature_{i}" for i in range(X.shape[1])]

            if X.shape[1] < 2: return "Not enough features for PCA (Need at least 2 numeric columns)."

            # Run PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            pca.fit(X_scaled)
            
            components = pca.components_
            var_ratio = pca.explained_variance_ratio_
            
            text = "=== CLUSTERING VISUALIZATION LOGIC (PCA) ===\n\n"
            text += "The 2D Cluster Plot is created by combining your features into 2 main components (PC1 and PC2).\n"
            text += "Here is what each component is made of:\n\n"
            
            for i, (comp, var) in enumerate(zip(components, var_ratio)):
                text += f"--- PC{i+1} (Explains {var:.1%} of variance) ---\n"
                text += f"PC{i+1} = "
                
                # Sort features by weight
                indices = np.argsort(np.abs(comp))[::-1]
                terms = []
                for idx in indices:
                    weight = comp[idx]
                    if abs(weight) > 0.1: # Only show significant features
                        terms.append(f"({weight:.2f} * {cols[idx]})")
                
                text += " + ".join(terms)
                text += "\n\n"
                
            return text
        except Exception as e:
            return f"Could not generate PCA text: {str(e)}"

    # --- PLOTTING IMPLEMENTATIONS ---

    def _plot_cluster_diagram_2d(self, output_dir):
        """Generates a 2D Scatter Plot of Clusters (Scaled & Robust)"""
        try:
            inner = self._unwrap_model()
            if not hasattr(inner, 'labels_'): return None
            
            X = self.train_data
            # Ensure we are using numeric data
            if isinstance(X, pd.DataFrame):
                X_num = X.select_dtypes(include=[np.number]).values
            else:
                X_num = X

            labels = inner.labels_
            plt.figure(figsize=(10, 8))
            
            # --- FIX: SCALE DATA BEFORE PLOTTING ---
            # PCA is sensitive to scale. We must scale to match the text output.
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_num)
            # ---------------------------------------

            # CASE 1: 1D Data
            if X_num.shape[1] == 1:
                plt.scatter(X_num[:, 0], np.zeros_like(X_num[:, 0]), c=labels, cmap='viridis', s=50, alpha=0.6)
                plt.title("Cluster Visualization (1D)")
                plt.xlabel("Feature Value")
                plt.yticks([])
                
            # CASE 2: Multi-Dimensional (Use PCA)
            else:
                pca = PCA(n_components=2)
                # Use the SCALED data
                X_2d = pca.fit_transform(X_scaled)
                
                scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
                
                # Labels should now match the text report (~36% and ~25%)
                plt.title("Cluster Visualization (PCA 2D Projection)")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} Var)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} Var)")
                plt.colorbar(scatter, label="Cluster ID")
                
            plt.grid(True, alpha=0.3)
            path = os.path.join(output_dir, "cluster_diagram_2d.png")
            plt.savefig(path)
            plt.close()
            return path
        except Exception as e:
            print(f"Cluster plot error: {e}")
            return None
    # ... (Keep other existing plotting methods like feature_importance, confusion_matrix, etc.) ...
    # For brevity, I'm assuming you kept the other standard methods from the previous working version.
    # If you need the FULL file again with ALL methods, let me know.
    
    def _plot_feature_importance(self, output_dir):
        try:
            model = self._unwrap_model()
            if not hasattr(model, 'feature_importances_'): return None
            importances = model.feature_importances_
            names = self.features
            if len(names) != len(importances): names = [f"F{i}" for i in range(len(importances))]
            indices = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [names[i] for i in indices], rotation=45, ha='right')
            plt.title("Feature Importance"); plt.tight_layout()
            path = os.path.join(output_dir, "feature_importance.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def plot_complete_decision_tree(self, output_dir, max_depth=None, class_names=None):
        try:
            plt.figure(figsize=(40, 20))
            inner = self._unwrap_model()
            if hasattr(inner, 'estimators_'): tree_obj = inner.estimators_[0]
            elif hasattr(inner, 'tree_'): tree_obj = inner
            else: return None
            feats = self.features
            if hasattr(tree_obj, "n_features_in_") and len(feats) != tree_obj.n_features_in_:
                feats = [f"F{i}" for i in range(tree_obj.n_features_in_)]
            plot_tree(tree_obj, feature_names=feats, class_names=class_names, filled=True, rounded=True, impurity=True, fontsize=10, max_depth=max_depth)
            path = os.path.join(output_dir, "decision_tree_viz.png")
            plt.savefig(path, bbox_inches='tight'); plt.close()
            return path
        except: return None

    def _plot_confusion_matrix(self, output_dir):
        try:
            y_true, y_pred = self._get_test_predictions()
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
            path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def _plot_roc_curve(self, output_dir):
        try:
            X_test, y_true = self.test_data
            y_proba = self.model.predict_proba(X_test)
            plt.figure(figsize=(10, 8))
            classes = np.unique(y_true)
            if len(classes) == 2:
                p = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
                fpr, tpr, _ = roc_curve(y_true, p, pos_label=classes[1])
                plt.plot(fpr, tpr, label='ROC')
            else:
                y_bin = label_binarize(y_true, classes=classes)
                for i in range(len(classes)):
                    if i < y_proba.shape[1]:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        plt.plot(fpr, tpr, label=f'Class {classes[i]}')
            plt.plot([0, 1], [0, 1], 'k--'); plt.title('ROC Curve'); plt.legend()
            path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def _plot_precision_recall_curve(self, output_dir):
        try:
            X_test, y_true = self.test_data
            y_proba = self.model.predict_proba(X_test)
            plt.figure(figsize=(10,8))
            classes = np.unique(y_true)
            if len(classes) == 2:
                p = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
                pr, rc, _ = precision_recall_curve(y_true, p, pos_label=classes[1])
                plt.plot(rc, pr, label='PR Curve')
            else:
                y_bin = label_binarize(y_true, classes=classes)
                for i in range(len(classes)):
                    if i < y_proba.shape[1]:
                        pr, rc, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                        plt.plot(rc, pr, label=f'Class {classes[i]}')
            plt.title('Precision-Recall Curve'); plt.legend()
            path = os.path.join(output_dir, "precision_recall_curve.png")
            plt.savefig(path); plt.close()
            return path
        except: return None
        
    def _plot_actual_vs_predicted_45(self, output_dir):
        try:
            y_true, y_pred = self._get_test_predictions()
            plt.figure(figsize=(10, 8))
            plt.scatter(y_true, y_pred, alpha=0.5)
            mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            plt.plot([mn, mx], [mn, mx], 'r--')
            plt.title("Actual vs Predicted"); path = os.path.join(output_dir, "actual_vs_predicted.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def _plot_residual_analysis(self, output_dir):
        try:
            y_true, y_pred = self._get_test_predictions()
            res = y_true - y_pred
            fig, ax = plt.subplots(2, 2, figsize=(16, 12))
            ax[0,0].scatter(y_pred, res, alpha=0.5); ax[0,0].axhline(0, color='r', linestyle='--'); ax[0,0].set_title("Res vs Pred")
            sns.histplot(res, kde=True, ax=ax[0,1]); ax[0,1].set_title("Dist")
            stats.probplot(res, dist="norm", plot=ax[1,0]); ax[1,0].set_title("Q-Q")
            ax[1,1].plot(res.values if hasattr(res,'values') else res, alpha=0.6); ax[1,1].set_title("Res vs Order")
            path = os.path.join(output_dir, "residual_analysis.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def _plot_learning_curve(self, output_dir):
        try:
            if self.train_data is None or self.target is None: return None
            X = self.train_data
            if isinstance(X, pd.DataFrame) and self.target in X.columns: X = X.drop(columns=[self.target])
            elif isinstance(X, tuple): X = X[0]
            y = self.results['y_true'] if 'y_true' in self.results else None
            if y is None: return None
            
            train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
            train_mean = np.mean(train_scores, axis=1); test_mean = np.mean(test_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', color="r", label="Train"); plt.plot(train_sizes, test_mean, 'o-', color="g", label="CV")
            plt.title("Learning Curve"); plt.legend(); plt.grid(True, alpha=0.3)
            path = os.path.join(output_dir, "learning_curve_overfitting.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def _plot_coefficients_equation(self, output_dir):
        try:
            inner = self._unwrap_model()
            if not hasattr(inner, 'coef_'): return None
            coefs = inner.coef_
            if coefs.ndim > 1: coefs = coefs[0]
            names = self.features
            if len(names) != len(coefs): names = [f"F{i}" for i in range(len(coefs))]
            df_coef = pd.DataFrame({'Feature': names, 'Weight': coefs}).sort_values('Weight', key=abs, ascending=False).head(15)
            plt.figure(figsize=(10, 8))
            sns.barplot(x="Weight", y="Feature", data=df_coef)
            plt.title("Coefficients"); path = os.path.join(output_dir, "coefficients.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def create_dynamic_comparison(self, output_dir, feature):
        try:
            if self.train_data is None: return None
            idx = self.features.index(feature) if feature in self.features else -1
            if idx == -1: return None
            X = self.train_data
            vals = X.iloc[:, idx].values if isinstance(X, pd.DataFrame) else X[:, idx]
            plt.figure(figsize=(10,6)); plt.hist(vals, bins=20, alpha=0.7); plt.title(f"Dist: {feature}")
            path = os.path.join(output_dir, f"feat_{feature}.png")
            plt.savefig(path); plt.close()
            return path
        except: return None

    def plot_polynomial_fit(self, output_dir, feature_name, custom_degree=2):
        """
        Generates a plot comparing Linear (Deg 1) vs Custom Polynomial fit.
        """
        try:
            if self.train_data is None or self.target is None: return None
            
            # 1. Get Data
            X = self.train_data
            if isinstance(X, pd.DataFrame):
                if feature_name not in X.columns: return None
                x_data = X[feature_name].values
                
                # Try to find target
                if self.target in X.columns:
                    y_data = X[self.target].values
                elif isinstance(self.train_data, pd.DataFrame) and self.target in self.results.get('train_data', pd.DataFrame()).columns:
                     y_data = self.results['train_data'][self.target].values
                else:
                    # Fallback for tuple data
                    return None 
            else:
                return None

            # Remove NaNs
            mask = ~np.isnan(x_data) & ~np.isnan(y_data)
            x_clean = x_data[mask]
            y_clean = y_data[mask]

            # Sort for clean plotting
            sort_idx = np.argsort(x_clean)
            x_sorted = x_clean[sort_idx]
            
            # 2. Fit Models
            # Linear (Baseline)
            z1 = np.polyfit(x_clean, y_clean, 1)
            p1 = np.poly1d(z1)
            
            # Custom Degree
            z_custom = np.polyfit(x_clean, y_clean, int(custom_degree))
            p_custom = np.poly1d(z_custom)

            # 3. Plot
            plt.figure(figsize=(12, 8))
            plt.scatter(x_clean, y_clean, alpha=0.3, color='gray', label='Actual Data')
            
            plt.plot(x_sorted, p1(x_sorted), 'r--', linewidth=2, label='Linear (Deg 1)')
            plt.plot(x_sorted, p_custom(x_sorted), 'b-', linewidth=3, label=f'Poly (Deg {custom_degree})')
            
            plt.title(f"Polynomial Analysis (Degree {custom_degree}): {feature_name} vs {self.target}", fontsize=16)
            plt.xlabel(feature_name)
            plt.ylabel(self.target)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            path = os.path.join(output_dir, f"poly_analysis_{feature_name}_deg{custom_degree}.png")
            plt.savefig(path)
            plt.close()
            return path
            
        except Exception as e:
            print(f"Poly plot error: {e}")
            return None
        