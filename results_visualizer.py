# results_visualizer.py - ENHANCED VERSION WITH COMPREHENSIVE GRAPH GENERATION
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (confusion_matrix, classification_report, 
                           mean_squared_error, r2_score, roc_curve, auc,
                           precision_recall_curve, precision_score, recall_score,
                           silhouette_score, calinski_harabasz_score, davies_bouldin_score)
import scipy.stats as stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AdvancedResultsVisualizer:
    def __init__(self, parent, results, problem_type, model, test_data=None, train_data=None):
        self.parent = parent
        self.results = results
        self.problem_type = problem_type
        self.model = model
        self.test_data = test_data
        self.train_data = train_data
        self.current_plots = {}
        self.generated_plots = {}
        
        # Store feature information
        self.features = results.get('features', [])
        self.target = results.get('target', None)
        
    def get_available_graphs(self):
        """Return all available graph types based on problem type and data"""
        available_graphs = {}
        
        if self.problem_type == "classification":
            available_graphs.update({
                "confusion_matrix": "Confusion Matrix",
                "roc_curve": "ROC Curve",
                "precision_recall_curve": "Precision-Recall Curve",
                "feature_importance": "Feature Importance",
                "class_distribution": "Class Distribution",
                "classification_report": "Classification Report Heatmap",
                "calibration_curve": "Calibration Curve",
                "lift_curve": "Lift Curve",
                "cumulative_gain": "Cumulative Gain Chart"
            })
            
        elif self.problem_type == "regression":
            available_graphs.update({
                "actual_vs_predicted": "Actual vs Predicted (45° line)",
                "residual_plot": "Residuals Plot",
                "qq_plot": "Q-Q Plot of Residuals",
                "feature_importance": "Feature Importance",
                "prediction_error": "Prediction Error Plot",
                "cooks_distance": "Cook's Distance",
                "leverage_plot": "Leverage Plot",
                "partial_dependence": "Partial Dependence Plots"
            })
            
        elif self.problem_type in ["clustering", "anomaly"]:
            available_graphs.update({
                "cluster_distribution": "Cluster Distribution",
                "silhouette_analysis": "Silhouette Analysis",
                "pca_visualization": "PCA Visualization",
                "tsne_visualization": "t-SNE Visualization",
                "elbow_curve": "Elbow Curve",
                "dendrogram": "Dendrogram (Hierarchical)",
                "pairplot_clusters": "Pairplot with Clusters"
            })
        
        # Add feature-target relationship graphs
        if self.target and self.features:
            available_graphs.update({
                "feature_vs_target": "Feature vs Target Scatter Plots",
                "correlation_heatmap": "Correlation Heatmap",
                "pairplot": "Pairplot of Features"
            })
            
        # Add model-specific graphs
        if hasattr(self.model, 'feature_importances_'):
            available_graphs["feature_importance_detailed"] = "Detailed Feature Importance"
            
        return available_graphs

    def generate_graph(self, graph_type, output_dir, **kwargs):
        """Generate a specific graph type"""
        try:
            graph_methods = {
                # Classification graphs
                "confusion_matrix": self._plot_confusion_matrix,
                "roc_curve": self._plot_roc_curve,
                "precision_recall_curve": self._plot_precision_recall_curve,
                "class_distribution": self._plot_class_distribution,
                "classification_report": self._plot_classification_report,
                "calibration_curve": self._plot_calibration_curve,
                
                # Regression graphs
                "actual_vs_predicted": self._plot_actual_vs_predicted,
                "residual_plot": self._plot_residuals,
                "qq_plot": self._plot_qq_residuals,
                "prediction_error": self._plot_prediction_error,
                "cooks_distance": self._plot_cooks_distance,
                
                # Clustering graphs
                "cluster_distribution": self._plot_cluster_distribution,
                "silhouette_analysis": self._plot_silhouette_analysis,
                "pca_visualization": self._plot_pca_visualization,
                "elbow_curve": self._plot_elbow_curve,
                "dendrogram": self._plot_dendrogram,
                
                # Feature analysis graphs
                "feature_importance": self._plot_feature_importance,
                "feature_importance_detailed": self._plot_detailed_feature_importance,
                "feature_vs_target": self._plot_feature_vs_target,
                "correlation_heatmap": self._plot_correlation_heatmap,
                "pairplot": self._plot_pairplot,
                "pairplot_clusters": self._plot_pairplot_clusters,
                
                # Additional comprehensive graphs
                "learning_curve": self._plot_learning_curve,
                "validation_curve": self._plot_validation_curve,
                "permutation_importance": self._plot_permutation_importance
            }
            
            if graph_type in graph_methods:
                plot_path = graph_methods[graph_type](output_dir, **kwargs)
                if plot_path:
                    graph_name = self.get_available_graphs().get(graph_type, graph_type)
                    self.generated_plots[graph_name] = plot_path
                    return plot_path
            else:
                messagebox.showerror("Error", f"Graph type '{graph_type}' not supported")
                return None
                
        except Exception as e:
            error_msg = f"Error generating {graph_type}: {str(e)}"
            messagebox.showerror("Graph Generation Error", error_msg)
            return None

    def generate_all_possible_graphs(self, output_dir):
        """Generate all possible graphs based on available data and model type"""
        all_plots = {}
        available_graphs = self.get_available_graphs()
        
        for graph_type, graph_name in available_graphs.items():
            try:
                plot_path = self.generate_graph(graph_type, output_dir)
                if plot_path:
                    all_plots[graph_name] = plot_path
                    print(f"✅ Generated: {graph_name}")
            except Exception as e:
                print(f"❌ Failed to generate {graph_name}: {str(e)}")
                continue
                
        return all_plots

    # ==================== CLASSIFICATION GRAPHS ====================
    def _plot_confusion_matrix(self, output_dir):
        """Plot confusion matrix with annotations"""
        if not self._has_test_predictions():
            return None
            
        y_true, y_pred = self._get_test_predictions()
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), 
                   yticklabels=np.unique(y_true))
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_roc_curve(self, output_dir):
        """Plot ROC curve for binary and multiclass classification"""
        if not self._has_test_predictions() or not hasattr(self.model, 'predict_proba'):
            return None
            
        try:
            X_test, y_true = self.test_data
            y_proba = self.model.predict_proba(X_test)
            
            plt.figure(figsize=(10, 8))
            
            if len(np.unique(y_true)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve', fontsize=16, fontweight='bold')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
            else:  # Multiclass classification
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc
                
                n_classes = len(np.unique(y_true))
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, 
                            label=f'Class {i} (AUC = {roc_auc:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Multiclass ROC Curve', fontsize=16, fontweight='bold')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"ROC curve error: {e}")
            return None

    def _plot_precision_recall_curve(self, output_dir):
        """Plot precision-recall curve"""
        if not self._has_test_predictions():
            return None
            
        try:
            X_test, y_true = self.test_data
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                return None
                
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = precision_score(y_true, self.model.predict(X_test))
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, lw=2, color='blue', 
                    label=f'Precision-Recall (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "precision_recall_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Precision-Recall curve error: {e}")
            return None

    def _plot_class_distribution(self, output_dir):
        """Plot class distribution"""
        if self.target and self.train_data is not None:
            y_data = self.train_data[self.target] if isinstance(self.train_data, pd.DataFrame) else self.train_data[1]
        else:
            return None
            
        plt.figure(figsize=(10, 6))
        unique_classes, class_counts = np.unique(y_data, return_counts=True)
        
        bars = plt.bar(unique_classes, class_counts, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{count}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_classification_report(self, output_dir):
        """Plot classification report as heatmap"""
        if not self._has_test_predictions():
            return None
            
        y_true, y_pred = self._get_test_predictions()
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='viridis', 
                   fmt='.3f', linewidths=0.5)
        plt.title('Classification Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "classification_report.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_calibration_curve(self, output_dir):
        """Plot calibration curve"""
        if not self._has_test_predictions() or not hasattr(self.model, 'predict_proba'):
            return None
            
        try:
            from sklearn.calibration import calibration_curve
            
            X_test, y_true = self.test_data
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10, strategy='uniform')
            
            plt.figure(figsize=(10, 8))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                    label="Model", color='blue')
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            plt.ylabel("Fraction of positives")
            plt.xlabel("Mean predicted value")
            plt.title('Calibration Curve', fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "calibration_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Calibration curve error: {e}")
            return None

    # ==================== REGRESSION GRAPHS ====================
    def _plot_actual_vs_predicted(self, output_dir):
        """Plot actual vs predicted with 45-degree line"""
        if not self._has_test_predictions():
            return None
            
        y_true, y_pred = self._get_test_predictions()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title('Actual vs Predicted Values\n(45° Line = Perfect Prediction)', 
                 fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_residuals(self, output_dir):
        """Plot residuals analysis"""
        if not self._has_test_predictions():
            return None
            
        y_true, y_pred = self._get_test_predictions()
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Residuals Analysis', fontsize=16, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        
        # Residuals over time/index
        axes[1, 1].plot(range(len(residuals)), residuals, alpha=0.6, color='purple')
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Sample Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "residuals_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_qq_residuals(self, output_dir):
        """Plot Q-Q plot of residuals"""
        if not self._has_test_predictions():
            return None
            
        y_true, y_pred = self._get_test_predictions()
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 8))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals\n(Normality Check)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "qq_plot_residuals.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_prediction_error(self, output_dir):
        """Plot prediction error"""
        if not self._has_test_predictions():
            return None
            
        y_true, y_pred = self._get_test_predictions()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction Error Plot', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add error statistics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        plt.text(0.05, 0.95, f'RMSE = {rmse:.4f}', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "prediction_error.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_cooks_distance(self, output_dir):
        """Plot Cook's distance for influential points"""
        if not self._has_test_predictions():
            return None
            
        try:
            from sklearn.linear_model import LinearRegression
            import statsmodels.api as sm
            
            X_test, y_true = self.test_data
            
            # Fit OLS model
            X_test_with_const = sm.add_constant(X_test)
            model = sm.OLS(y_true, X_test_with_const).fit()
            
            # Get influence measures
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]
            
            plt.figure(figsize=(12, 6))
            plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
            plt.axhline(y=4/len(cooks_d), color='red', linestyle='--', 
                       label='4/n threshold')
            plt.xlabel('Sample Index')
            plt.ylabel("Cook's Distance")
            plt.title("Cook's Distance for Influential Points", 
                     fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "cooks_distance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Cook's distance error: {e}")
            return None

    # ==================== CLUSTERING GRAPHS ====================
    def _plot_cluster_distribution(self, output_dir):
        """Plot cluster distribution"""
        if not hasattr(self.model, 'labels_'):
            return None
            
        labels = self.model.labels_
        
        plt.figure(figsize=(10, 6))
        unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
        
        bars = plt.bar(unique_clusters, cluster_counts, color='lightcoral', 
                      edgecolor='black', alpha=0.7)
        plt.title('Cluster Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Points')
        plt.xticks(unique_clusters)
        
        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{count}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "cluster_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_silhouette_analysis(self, output_dir):
        """Plot silhouette analysis for clustering"""
        if not hasattr(self.model, 'labels_') or self.train_data is None:
            return None
            
        try:
            from sklearn.metrics import silhouette_samples, silhouette_score
            
            X = self.train_data[0] if isinstance(self.train_data, tuple) else self.train_data
            labels = self.model.labels_
            
            silhouette_avg = silhouette_score(X, labels)
            sample_silhouette_values = silhouette_samples(X, labels)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Silhouette Analysis\n(Average Score: {silhouette_avg:.3f})', 
                        fontsize=16, fontweight='bold')
            
            y_lower = 10
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                ith_cluster_silhouette_values = sample_silhouette_values[labels == label]
                ith_cluster_silhouette_values.sort()
                
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = plt.cm.nipy_spectral(float(i) / len(unique_labels))
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
                y_lower = y_upper + 10
            
            ax1.set_xlabel("Silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            
            # Second plot: actual clusters
            colors = plt.cm.nipy_spectral(labels.astype(float) / len(unique_labels))
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                       c=colors, edgecolor='k')
            ax2.set_xlabel("Feature 1")
            ax2.set_ylabel("Feature 2")
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "silhouette_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Silhouette analysis error: {e}")
            return None

    def _plot_pca_visualization(self, output_dir):
        """Plot PCA visualization of clusters"""
        if not hasattr(self.model, 'labels_') or self.train_data is None:
            return None
            
        try:
            from sklearn.decomposition import PCA
            
            X = self.train_data[0] if isinstance(self.train_data, tuple) else self.train_data
            labels = self.model.labels_
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                cmap='viridis', alpha=0.7, edgecolors='k')
            plt.colorbar(scatter)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Visualization of Clusters', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "pca_visualization.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"PCA visualization error: {e}")
            return None

    def _plot_elbow_curve(self, output_dir):
        """Plot elbow curve for K-means (if applicable)"""
        if not hasattr(self.model, 'inertia_') or self.train_data is None:
            return None
            
        try:
            from sklearn.cluster import KMeans
            
            X = self.train_data[0] if isinstance(self.train_data, tuple) else self.train_data
            
            # Calculate inertia for different k values
            k_range = range(1, 11)
            inertias = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Curve for Optimal k', fontsize=16, fontweight='bold')
            plt.xticks(k_range)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "elbow_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Elbow curve error: {e}")
            return None

    def _plot_dendrogram(self, output_dir):
        """Plot dendrogram for hierarchical clustering"""
        if not hasattr(self.model, 'labels_') or self.train_data is None:
            return None
            
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from sklearn.cluster import AgglomerativeClustering
            
            X = self.train_data[0] if isinstance(self.train_data, tuple) else self.train_data
            
            # Use a sample for large datasets
            if len(X) > 1000:
                indices = np.random.choice(len(X), 1000, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Create linkage matrix
            Z = linkage(X_sample, method='ward')
            
            plt.figure(figsize=(15, 8))
            dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True)
            plt.xlabel('Sample Index or Cluster Size')
            plt.ylabel('Distance')
            plt.title('Dendrogram for Hierarchical Clustering', 
                     fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "dendrogram.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Dendrogram error: {e}")
            return None

    # ==================== FEATURE ANALYSIS GRAPHS ====================
    def _plot_feature_importance(self, output_dir):
        """Plot basic feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.title('Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importances)), [self.features[i] for i in indices], 
                 rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_detailed_feature_importance(self, output_dir):
        """Plot detailed feature importance with values"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(importances)), importances[indices], 
                      color='skyblue', edgecolor='black')
        plt.title('Detailed Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(importances)), [self.features[i] for i in indices], 
                 rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances[indices]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "detailed_feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

    def _plot_feature_vs_target(self, output_dir):
        """Plot feature vs target relationships"""
        if self.target is None or not self.features:
            return None
            
        try:
            # Create subplots for multiple features
            n_features = min(len(self.features), 9)  # Limit to 9 features
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            fig.suptitle('Feature vs Target Relationships', fontsize=16, fontweight='bold')
            
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(self.features[:n_features]):
                if i < len(axes):
                    if pd.api.types.is_numeric_dtype(self.train_data[feature]):
                        # Scatter plot for numeric features
                        axes[i].scatter(self.train_data[feature], self.train_data[self.target], 
                                      alpha=0.6, s=30)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel(self.target)
                    else:
                        # Box plot for categorical features
                        data_to_plot = []
                        unique_cats = self.train_data[feature].unique()[:10]  # Limit categories
                        for cat in unique_cats:
                            data_to_plot.append(self.train_data[self.train_data[feature] == cat][self.target])
                        
                        axes[i].boxplot(data_to_plot, labels=unique_cats)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel(self.target)
                        axes[i].tick_params(axis='x', rotation=45)
                    
                    axes[i].set_title(f'{feature} vs {self.target}')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "feature_vs_target.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Feature vs target plot error: {e}")
            return None

    def _plot_correlation_heatmap(self, output_dir):
        """Plot correlation heatmap"""
        if self.train_data is None or not isinstance(self.train_data, pd.DataFrame):
            return None
            
        try:
            # Select only numeric columns
            numeric_data = self.train_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return None
                
            corr_matrix = numeric_data.corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Correlation heatmap error: {e}")
            return None

    def _plot_pairplot(self, output_dir):
        """Plot pairplot of features"""
        if self.train_data is None or not isinstance(self.train_data, pd.DataFrame):
            return None
            
        try:
            # Select top 5 numeric features for pairplot
            numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
            if self.target and self.target in numeric_cols:
                numeric_cols.remove(self.target)
                
            selected_features = numeric_cols[:5]  # Limit to 5 features
            if self.target:
                selected_features.append(self.target)
                
            if len(selected_features) < 2:
                return None
                
            pairplot_data = self.train_data[selected_features]
            
            # Use seaborn pairplot
            g = sns.pairplot(pairplot_data, diag_kind='hist', 
                           plot_kws={'alpha': 0.6, 's': 30})
            g.fig.suptitle('Feature Pairplot', fontsize=16, fontweight='bold', y=1.02)
            
            plot_path = os.path.join(output_dir, "pairplot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Pairplot error: {e}")
            return None

    def _plot_pairplot_clusters(self, output_dir):
        """Plot pairplot with cluster coloring"""
        if not hasattr(self.model, 'labels_') or self.train_data is None:
            return None
            
        try:
            X = self.train_data[0] if isinstance(self.train_data, tuple) else self.train_data
            labels = self.model.labels_
            
            # Convert to DataFrame for pairplot
            if isinstance(self.train_data, pd.DataFrame):
                plot_data = self.train_data.copy()
            else:
                # Create DataFrame from numpy array
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
                plot_data = pd.DataFrame(X, columns=feature_names)
            
            plot_data['Cluster'] = labels
            
            # Select top 4 features for pairplot
            numeric_cols = plot_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Cluster' in numeric_cols:
                numeric_cols.remove('Cluster')
            selected_features = numeric_cols[:4]
            
            if len(selected_features) < 2:
                return None
                
            g = sns.pairplot(plot_data[selected_features + ['Cluster']], 
                           hue='Cluster', palette='viridis', 
                           plot_kws={'alpha': 0.7, 's': 30})
            g.fig.suptitle('Pairplot with Cluster Coloring', 
                          fontsize=16, fontweight='bold', y=1.02)
            
            plot_path = os.path.join(output_dir, "pairplot_clusters.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Pairplot clusters error: {e}")
            return None

    # ==================== ADDITIONAL GRAPHS ====================
    def _plot_learning_curve(self, output_dir):
        """Plot learning curve"""
        if self.train_data is None:
            return None
            
        try:
            from sklearn.model_selection import learning_curve
            
            X = self.train_data[0] if isinstance(self.train_data, tuple) else self.train_data
            y = self.train_data[1] if isinstance(self.train_data, tuple) else self.train_data[self.target]
            
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, X, y, cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10))
            
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                           test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.title("Learning Curve", fontsize=16, fontweight='bold')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "learning_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Learning curve error: {e}")
            return None

    def _plot_validation_curve(self, output_dir):
        """Plot validation curve for hyperparameter tuning"""
        # This would require knowing which hyperparameters to tune
        # For now, return None as it's model-specific
        return None

    def _plot_permutation_importance(self, output_dir):
        """Plot permutation importance"""
        if not self._has_test_predictions():
            return None
            
        try:
            from sklearn.inspection import permutation_importance
            
            X_test, y_true = self.test_data
            result = permutation_importance(self.model, X_test, y_true, 
                                          n_repeats=10, random_state=42)
            
            sorted_idx = result.importances_mean.argsort()[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.boxplot(result.importances[sorted_idx].T,
                       labels=[self.features[i] for i in sorted_idx])
            plt.xticks(rotation=45, ha='right')
            plt.title('Permutation Importance', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "permutation_importance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Permutation importance error: {e}")
            return None

    # ==================== HELPER METHODS ====================
    def _has_test_predictions(self):
        """Check if test predictions are available"""
        return (self.test_data is not None and 
                'y_true' in self.results and 
                'y_pred' in self.results)

    def _get_test_predictions(self):
        """Get test predictions"""
        return self.results['y_true'], self.results['y_pred']

    def get_generated_plots(self):
        """Get all generated plots"""
        return self.generated_plots

    def clear_generated_plots(self):
        """Clear generated plots cache"""
        self.generated_plots = {}

    def create_comprehensive_dashboard(self, output_dir):
        """Compatibility method for existing code - uses the new generate_all_possible_graphs"""
        return self.generate_all_possible_graphs(output_dir)

    def get_available_graphs(self):
        """Return only supported graph types based on available data"""
        available_graphs = {}
        
        # Basic graphs that should always work
        base_graphs = {
            "feature_importance": "Feature Importance",
            "metrics_summary": "Metrics Summary"
        }
        
        if self.problem_type == "classification":
            available_graphs.update(base_graphs)
            if self._has_test_predictions():
                available_graphs.update({
                    "confusion_matrix": "Confusion Matrix",
                    "roc_curve": "ROC Curve",
                })
            if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                available_graphs["decision_tree"] = "Decision Tree"
                
        elif self.problem_type == "regression":
            available_graphs.update(base_graphs)
            if self._has_test_predictions():
                available_graphs.update({
                    "actual_vs_predicted": "Actual vs Predicted",
                    "residual_analysis": "Residual Analysis",
                })
                
        elif self.problem_type in ["clustering", "anomaly"]:
            available_graphs.update(base_graphs)
            if hasattr(self.model, 'labels_'):
                available_graphs.update({
                    "cluster_diagram": "Cluster Diagram",
                    "pca_clusters": "PCA Visualization"
                })
        
        # Remove unsupported graphs
        unsupported = ["leverage_plot", "partial_dependence", "cooks_distance"]
        for graph in unsupported:
            available_graphs.pop(graph, None)
        
        return available_graphs


# Keep the original class for backward compatibility
class ResultsVisualizer(AdvancedResultsVisualizer):
    pass