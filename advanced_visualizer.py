# advanced_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
import scipy.stats as stats

class AdvancedVisualizer:
    def __init__(self, results, problem_type, model, test_data=None):
        self.results = results
        self.problem_type = problem_type
        self.model = model
        self.test_data = test_data
        
    def create_comprehensive_plots(self, output_dir):
        """Create all relevant plots based on problem type"""
        plots_created = []
        
        if self.problem_type in ["classification", "regression"]:
            # Common plots for both
            plots_created.extend(self._create_common_plots(output_dir))
            
            if self.problem_type == "classification":
                plots_created.extend(self._create_classification_plots(output_dir))
            else:
                plots_created.extend(self._create_regression_plots(output_dir))
                
        return plots_created
        
    def _create_common_plots(self, output_dir):
        """Create plots common to both classification and regression"""
        plots = []
        
        try:
            # Feature importance plot
            if hasattr(self.model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                features = self.results.get('features', [])
                importances = self.model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                plt.title("Feature Importances")
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
                plt.tight_layout()
                
                plot_path = os.path.join(output_dir, "feature_importance.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots.append(plot_path)
                
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
            
        return plots
        
    def _create_regression_plots(self, output_dir):
        """Create regression-specific plots"""
        plots = []
        
        try:
            # Actual vs Predicted scatter plot
            if 'y_true' in self.results and 'y_pred' in self.results:
                y_true = self.results['y_true']
                y_pred = self.results['y_pred']
                
                plt.figure(figsize=(10, 8))
                
                # Scatter plot
                plt.subplot(2, 2, 1)
                plt.scatter(y_true, y_pred, alpha=0.6)
                plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted')
                
                # Residuals plot
                plt.subplot(2, 2, 2)
                residuals = y_true - y_pred
                plt.scatter(y_pred, residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                plt.title('Residuals Plot')
                
                # Distribution of residuals
                plt.subplot(2, 2, 3)
                plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Residuals')
                plt.ylabel('Frequency')
                plt.title('Distribution of Residuals')
                
                # Q-Q plot
                plt.subplot(2, 2, 4)
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Q-Q Plot of Residuals')
                
                plt.tight_layout()
                
                plot_path = os.path.join(output_dir, "regression_diagnostics.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots.append(plot_path)
                
                # Metrics comparison
                self._create_metrics_plot(output_dir, plots)
                
        except Exception as e:
            print(f"Could not create regression plots: {e}")
            
        return plots
        
    def _create_classification_plots(self, output_dir):
        """Create classification-specific plots"""
        plots = []
        
        try:
            if 'y_true' in self.results and 'y_pred' in self.results:
                y_true = self.results['y_true']
                y_pred = self.results['y_pred']
                
                # Confusion Matrix
                plt.figure(figsize=(10, 8))
                
                # Confusion matrix
                plt.subplot(2, 2, 1)
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                
                # Classification report heatmap
                plt.subplot(2, 2, 2)
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='viridis')
                plt.title('Classification Report')
                
                # ROC Curve (for binary classification)
                if len(np.unique(y_true)) == 2:
                    plt.subplot(2, 2, 3)
                    from sklearn.metrics import roc_curve, auc
                    if hasattr(self.model, 'predict_proba'):
                        y_score = self.model.predict_proba(self.test_data[0])[:, 1]
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('ROC Curve')
                        plt.legend(loc="lower right")
                
                plt.tight_layout()
                
                plot_path = os.path.join(output_dir, "classification_analysis.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots.append(plot_path)
                
        except Exception as e:
            print(f"Could not create classification plots: {e}")
            
        return plots
        
    def _create_metrics_plot(self, output_dir, plots):
        """Create metrics comparison plot"""
        try:
            metrics = self.results.get('metrics', {})
            if metrics:
                plt.figure(figsize=(10, 6))
                
                metric_names = []
                metric_values = []
                
                for metric, value in metrics.items():
                    if metric not in ['run_log', 'test_predictions_made']:
                        metric_names.append(metric)
                        metric_values.append(value)
                
                if metric_names:
                    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
                    bars = plt.bar(metric_names, metric_values, color=colors)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metric_values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.4f}', ha='center', va='bottom')
                    
                    plt.title('Model Performance Metrics')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(output_dir, "metrics_comparison.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots.append(plot_path)
                    
        except Exception as e:
            print(f"Could not create metrics plot: {e}")