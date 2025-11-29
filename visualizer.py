import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import imageio

class Visualizer:
    def __init__(self, df, features, target, trainer):
        self.df = df
        self.features = features
        self.target = target
        self.trainer = trainer

    def plot_and_save(self, results, out_dir, fmt="png"):
        os.makedirs(out_dir, exist_ok=True)
        basename = f"fit_{self.target}_{'_'.join(self.features)}"
        if fmt.lower() in ("png","jpeg","jpg"):
            outpath = os.path.join(out_dir, basename + "." + (fmt if fmt!="jpg" else "jpeg"))
            self._plot_to_file(results, outpath)
            return outpath
        elif fmt.lower()=="gif":
            # simple gif: animate predicted vs actual gradually
            frames = []
            tmp_files = []
            preds = results.get("preds")
            if preds is None:
                preds = results.get("y")
            y = results.get("y_test") if "y_test" in results else results.get("y") if "y" in results else results.get("y_train")
            X_plot = results.get("X_test") if "X_test" in results else results.get("X")
            # We'll animate by plotting first k points
            for k in range(5, len(preds)+1, max(1, len(preds)//20)):
                fname = os.path.join(out_dir, f"_tmp_{k}.png")
                self._plot_to_file(results, fname, upto=k)
                tmp_files.append(fname)
            gif_path = os.path.join(out_dir, basename + ".gif")
            images = [imageio.imread(f) for f in tmp_files]
            imageio.mimsave(gif_path, images, fps=3)
            # cleanup tmp
            for f in tmp_files:
                os.remove(f)
            return gif_path
        else:
            raise ValueError("Unsupported format")

    def _plot_to_file(self, results, outpath, upto=None):
        # If single feature, show scatter + fit line. If multi-feature, show true vs predicted.
        preds = results.get("preds")
        if upto is None:
            upto = len(preds)
        plt.figure(figsize=(8,6))
        if len(self.features) == 1:
            x_col = self.features[0]
            xs = self.df[x_col].values
            # if train_test, use y_test & preds
            y_true = results.get("y_test") if "y_test" in results else results.get("y") if "y" in results else results.get("y_train")
            if y_true is None:
                y_true = self.df[self.target].values
            # scatter
            plt.scatter(xs[:upto], y_true[:upto], label="actual", alpha=0.6)
            plt.scatter(xs[:upto], preds[:upto], label="predicted", alpha=0.6)
            # sort for line plot if single feature and polynomial transformation used
            try:
                order = np.argsort(xs)
                xs_sorted = xs[order]
                preds_sorted = preds[order]
                plt.plot(xs_sorted, preds_sorted, linewidth=2, label="fit line")
            except Exception:
                pass
            plt.xlabel(x_col)
            plt.ylabel(self.target)
            plt.legend()
            plt.title("Actual vs Predicted")
        else:
            # Multi-feature: plot predicted vs actual scatter
            y_true = results.get("y_test") if "y_test" in results else results.get("y") if "y" in results else self.df[self.target].values
            plt.scatter(y_true[:upto], preds[:upto], alpha=0.7)
            minv = min(min(y_true[:upto]), min(preds[:upto]))
            maxv = max(max(y_true[:upto]), max(preds[:upto]))
            plt.plot([minv,maxv],[minv,maxv], linestyle='--', label='ideal')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            plt.legend()
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        
        
    def plot_decision_tree(self, output_dir, tree_index=0):
        """Plot decision tree for tree-based models"""
        try:
            if hasattr(self.model, 'estimators_'):
                # For Random Forest
                from sklearn.tree import plot_tree
                plt.figure(figsize=(20, 10))
                plot_tree(self.model.estimators_[tree_index], 
                        feature_names=self.results.get('features', []),
                        filled=True, rounded=True)
                plot_path = os.path.join(output_dir, f"decision_tree_{tree_index}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return plot_path
        except Exception as e:
            print(f"Could not plot decision tree: {e}")
        return None

    # Add this to your visualizer.py or enhanced_visualizer.py

    def plot_complete_decision_tree(self, output_dir, max_depth=None):
        """Plot complete decision tree with proper sizing"""
        try:
            from sklearn.tree import plot_tree, export_text, export_graphviz
            import graphviz
            
            plt.figure(figsize=(40, 20))
            
            if hasattr(self.model, 'estimators_'):  # Random Forest
                tree_to_plot = self.model.estimators_[0]
                tree_name = "Random_Forest_Tree_0"
            elif hasattr(self.model, 'tree_'):  # Single Decision Tree
                tree_to_plot = self.model
                tree_name = "Decision_Tree"
            else:
                return None
            
            # Plot with better parameters
            plot_tree(tree_to_plot,
                    feature_names=self.features,
                    filled=True,
                    rounded=True,
                    proportion=True,
                    impurity=False,
                    node_ids=True,
                    fontsize=8,
                    max_depth=max_depth)
            
            plt.title(f'Decision Tree - {tree_name}', fontsize=16, pad=20)
            plot_path = os.path.join(output_dir, f"{tree_name}_complete.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Also generate text rules
            rules_path = self._export_decision_rules(output_dir, tree_to_plot)
            
            return plot_path
            
        except Exception as e:
            print(f"Decision tree plotting error: {e}")
            return None

    def _export_decision_rules(self, output_dir, tree):
        """Export decision rules as text and boolean logic"""
        try:
            from sklearn.tree import export_text
            
            # Traditional text rules
            rules_text = export_text(tree, feature_names=self.features)
            
            # Enhanced boolean rules
            boolean_rules = self._extract_boolean_rules(tree)
            
            # Save to file
            rules_path = os.path.join(output_dir, "decision_rules.txt")
            with open(rules_path, 'w') as f:
                f.write("DECISION TREE RULES\n")
                f.write("=" * 50 + "\n\n")
                f.write("TEXT FORMAT:\n")
                f.write(rules_text)
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("BOOLEAN LOGIC FORMAT:\n")
                f.write(boolean_rules)
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("MATHEMATICAL DECISION PATH:\n")
                f.write(self._extract_decision_paths(tree))
            
            return rules_path
            
        except Exception as e:
            print(f"Rules extraction error: {e}")
            return None

    def _extract_boolean_rules(self, tree):
        """Extract rules in boolean logic format"""
        try:
            from sklearn.tree import _tree
            
            tree_ = tree.tree_
            feature_names = self.features
            boolean_rules = []
            
            def recurse(node, depth, rule):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_names[tree_.feature[node]]
                    threshold = tree_.threshold[node]
                    
                    # Left child rule
                    left_rule = f"{rule} AND ({name} <= {threshold:.4f})" if rule else f"({name} <= {threshold:.4f})"
                    recurse(tree_.children_left[node], depth + 1, left_rule)
                    
                    # Right child rule  
                    right_rule = f"{rule} AND ({name} > {threshold:.4f})" if rule else f"({name} > {threshold:.4f})"
                    recurse(tree_.children_right[node], depth + 1, right_rule)
                else:
                    # Leaf node
                    value = tree_.value[node][0]
                    pred_class = value.argmax()
                    boolean_rules.append(f"IF {rule} THEN Class = {pred_class} (samples: {int(np.sum(value))})")
            
            recurse(0, 1, "")
            return "\n".join(boolean_rules[:50])  # Limit output
            
        except Exception as e:
            return f"Boolean rules extraction failed: {str(e)}"

    def _extract_decision_paths(self, tree):
        """Extract decision paths in mathematical form"""
        try:
            paths = []
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
            
            while len(stack) > 0:
                node_id, depth = stack.pop()
                node_depth[node_id] = depth

                if (children_left[node_id] != children_right[node_id]):
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True
            
            for i in range(n_nodes):
                if is_leaves[i]:
                    path = f"Node {i}: LEAF → Prediction = {tree.tree_.value[i].argmax()}"
                    paths.append(path)
                else:
                    path = f"Node {i}: IF {self.features[feature[i]]} <= {threshold[i]:.4f}"
                    paths.append(path)
            
            return "\n".join(paths[:30])  # Limit output
            
        except Exception as e:
            return f"Decision path extraction failed: {str(e)}"