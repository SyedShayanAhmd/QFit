import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import tkinter as tk
from tkinter import ttk, messagebox

class AdvancedPreprocessor:
    def __init__(self, df):
        self.df_original = df.copy()
        self.processed_df = None
        self.transformations_log = []
        
    # In your preprocess.py, modify the _apply_dtype_conversion method:

# UPDATE IN preprocess.py - ENHANCE THE _apply_dtype_conversion METHOD

    def _apply_dtype_conversion(self, df, col, settings):
        """Convert column to specified data type - ENHANCED VERSION"""
        target_dtype = settings.get("dtype_conversion", "").strip().lower()
        
        if not target_dtype or col not in df.columns:
            return df
            
        original_dtype = str(df[col].dtype)
        
        try:
            # Create a working copy
            series = df[col].copy()
            
            # Enhanced cleaning for object columns
            if series.dtype == 'object':
                series = series.astype(str)
                
                # More comprehensive cleaning
                series = series.str.strip()
                
                # Handle common data issues
                series = series.replace(['', 'nan', 'None', 'null', 'NaN', 'N/A', 'n/a'], np.nan)
                
                # Remove common non-numeric characters for numeric conversion
                if target_dtype in ["int", "integer", "float", "float64"]:
                    series = series.str.replace(r'[^\d.-]', '', regex=True)
            
            conversion_success = True
            
            if target_dtype in ["int", "integer", "int64"]:
                # Use pandas nullable integer type
                numeric_series = pd.to_numeric(series, errors='coerce')
                if not numeric_series.isna().all():
                    df[col] = numeric_series.astype('Int64')
                else:
                    conversion_success = False
                    
            elif target_dtype in ["float", "float64"]:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if not numeric_series.isna().all():
                    df[col] = numeric_series.astype(float)
                else:
                    conversion_success = False
                    
            elif target_dtype in ["bool", "boolean"]:
                # Enhanced boolean conversion
                true_values = ['true', 'yes', 'y', '1', 't', 'yes', 'on', 'enable']
                false_values = ['false', 'no', 'n', '0', 'f', 'no', 'off', 'disable']
                
                series = series.str.lower().str.strip()
                series = series.replace(true_values, True)
                series = series.replace(false_values, False)
                series = series.replace(['', 'nan'], np.nan)
                
                if series.notna().any():
                    df[col] = series.astype('boolean')
                else:
                    conversion_success = False
                    
            elif target_dtype in ["str", "string", "object"]:
                df[col] = series.astype(str)
                
            elif target_dtype in ["category", "categorical"]:
                if series.notna().any():
                    df[col] = series.astype('category')
                else:
                    conversion_success = False
            
            # Log conversion results
            if conversion_success:
                new_dtype = str(df[col].dtype)
                converted_count = df[col].notna().sum()
                total_count = len(df[col])
                conversion_rate = (converted_count / total_count) * 100
                
                self._log_transformation(
                    f"Column '{col}': {original_dtype} → {new_dtype} "
                    f"({conversion_rate:.1f}% success, {converted_count}/{total_count} values)"
                )
            else:
                self._log_transformation(
                    f"Column '{col}': Failed to convert {original_dtype} → {target_dtype}"
                )
                
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error converting to {target_dtype} - {str(e)}")
            
        return df
      
    def apply_advanced_transformations(self, column_settings, global_settings, selected_features=None, target=None):
        """Apply all transformations including advanced settings"""
        df = self.df_original.copy()
        self.transformations_log = []
        
        # Column selection
        if selected_features or target:
            cols = []
            if target and target in df.columns:
                cols.append(target)
            if selected_features:
                for f in selected_features:
                    if f in df.columns and f not in cols:
                        cols.append(f)
            if cols:
                df = df[cols]
                self._log_transformation(f"Selected columns: {', '.join(cols)}")
        
        # Apply per-column transformations
        for col, settings in column_settings.items():
            if col not in df.columns:
                continue
                
            df = self._apply_column_transformations(df, col, settings)
        
        if target:
            global_settings['target_col'] = target
        
        # Apply global transformations
        df = self._apply_global_transformations(df, global_settings)
        
        self.processed_df = df
        return df
    
    def _apply_column_transformations(self, df, col, settings):
        """Apply all transformations for a single column"""
        original_dtype = str(df[col].dtype)
        
        # Row filtering by value (do this FIRST)
        df = self._apply_row_filtering(df, col, settings)
        
        # If column was removed by filtering (shouldn't happen but safe check)
        if col not in df.columns:
            return df
        
        # Missing values
        df = self._handle_missing_values(df, col, settings)
        
        # If column was removed by missing values handling
        if col not in df.columns:
            return df
        
        # Outlier removal
        df = self._handle_outliers(df, col, settings)
        
        # Data type conversion (add this here)
        df = self._apply_dtype_conversion(df, col, settings)
        
        # Custom mapping
        df = self._apply_custom_mapping(df, col, settings)
        
        # Strip substring
        strip_text = settings.get("strip_substring", "").strip()
        if strip_text and col in df.columns:
            df[col] = df[col].astype(str).str.replace(strip_text, "", case=False, regex=False).str.strip()
            self._log_transformation(f"{col}: stripped substring '{strip_text}'")
        
        # Binning
        df = self._apply_binning(df, col, settings)
        
        # Pattern groups
        df = self._apply_pattern_groups(df, col, settings)
        
        # Clamping
        df = self._apply_clamping(df, col, settings)
        
        # Percentage rules
        df = self._apply_percentage_rules(df, col, settings)
        
        # Encoding (do this LAST since it can remove the original column)
        df = self._apply_encoding(df, col, settings)
        
        # Log changes
        if col in df.columns:
            new_dtype = str(df[col].dtype)
            if original_dtype != new_dtype:
                self._log_transformation(f"Column '{col}': {original_dtype} → {new_dtype}")
                
        return df

    def _apply_row_filtering(self, df, col, settings):
        """Remove rows based on value patterns - but make it reversible"""
        remove_values = settings.get("remove_values", "").strip()
        remove_mode = settings.get("remove_mode", "equals")
        
        if not remove_values:
            return df
            
        # Check if column still exists (might have been removed by previous operations)
        if col not in df.columns:
            return df
            
        values_to_remove = [v.strip() for v in remove_values.split(",") if v.strip()]
        if not values_to_remove:
            return df
            
        original_count = len(df)
        series = df[col].astype(str).fillna("")
        
        # Create mask based on removal mode
        if remove_mode == "equals":
            mask = ~series.isin(values_to_remove)
        elif remove_mode == "contains":
            mask = ~series.str.contains('|'.join(values_to_remove), case=False, na=False)
        elif remove_mode == "startswith":
            pattern = '|'.join([f"^{re.escape(v)}" for v in values_to_remove])
            mask = ~series.str.contains(pattern, case=False, na=False)
        elif remove_mode == "endswith":
            pattern = '|'.join([f"{re.escape(v)}$" for v in values_to_remove])
            mask = ~series.str.contains(pattern, case=False, na=False)
        elif remove_mode == "regex":
            try:
                pattern = '|'.join(values_to_remove)
                mask = ~series.str.contains(pattern, case=False, na=False)
            except:
                mask = pd.Series(True, index=df.index)
        else:
            mask = pd.Series(True, index=df.index)
        
        df = df[mask]
        removed_count = original_count - len(df)
        
        if removed_count > 0:
            self._log_transformation(f"Column '{col}': Removed {removed_count} rows matching {values_to_remove} (mode: {remove_mode})")
        
        return df
    
    def _handle_missing_values(self, df, col, settings):
        """Handle missing values with various strategies"""
        strat = settings.get("strategy", "none")
        
        if strat == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].mean())
                self._log_transformation(f"Column '{col}': Filled {missing_count} missing values with mean")
                
        elif strat == "median" and pd.api.types.is_numeric_dtype(df[col]):
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].median())
                self._log_transformation(f"Column '{col}': Filled {missing_count} missing values with median")
                
        elif strat == "mode":
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                self._log_transformation(f"Column '{col}': Filled {missing_count} missing values with mode '{fill_val}'")
                
        elif strat == "fill_value":
            fill_val = settings.get("fill_value", "")
            if fill_val:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(fill_val)
                    self._log_transformation(f"Column '{col}': Filled {missing_count} missing values with '{fill_val}'")
                    
        elif strat == "drop_row":
            original_rows = len(df)
            df = df[df[col].notna()]
            removed_rows = original_rows - len(df)
            if removed_rows > 0:
                self._log_transformation(f"Column '{col}': Removed {removed_rows} rows with missing values")
                
        return df
    
    def _handle_outliers(self, df, col, settings):
        """Remove outliers based on min/max values"""
        low = settings.get("low", "").strip()
        high = settings.get("high", "").strip()
        
        if not low and not high:
            return df
            
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            keep_mask = pd.Series(True, index=df.index)
            
            if low:
                low_val = float(low)
                keep_mask = keep_mask & (numeric_series >= low_val)
                
            if high:
                high_val = float(high)
                keep_mask = keep_mask & (numeric_series <= high_val)
            
            removed_count = (~keep_mask).sum()
            if removed_count > 0:
                df = df[keep_mask]
                range_text = []
                if low: range_text.append(f">= {low}")
                if high: range_text.append(f"<= {high}")
                self._log_transformation(f"Column '{col}': Removed {removed_count} outliers outside {' and '.join(range_text)}")
                
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in outlier removal - {str(e)}")
            
        return df
    
    def _apply_custom_mapping(self, df, col, settings):
        """Apply custom value mappings"""
        mapping_str = settings.get("custom_mapping", "").strip()
        if not mapping_str:
            return df
            
        try:
            mapping = {}
            for item in mapping_str.split(","):
                if ":" in item:
                    key, value = item.split(":", 1)
                    mapping[key.strip()] = value.strip()
            
            if mapping:
                original_unique = df[col].nunique()
                df[col] = df[col].astype(str).replace(mapping)
                new_unique = df[col].nunique()
                self._log_transformation(f"Column '{col}': Applied custom mapping ({original_unique} → {new_unique} unique values)")
                
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in custom mapping - {str(e)}")
            
        return df
    
    def _apply_binning(self, df, col, settings):
        """Apply numeric binning"""
        bins_str = settings.get("bins", "").strip()
        labels_str = settings.get("bin_labels", "").strip()
        
        if not bins_str or not labels_str:
            return df
            
        try:
            bins = [float(x.strip()) for x in bins_str.split(",") if x.strip()]
            labels = [x.strip() for x in labels_str.split(",") if x.strip()]
            
            if len(bins) > 1 and len(labels) == len(bins) - 1:
                df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
                self._log_transformation(f"Column '{col}': Applied binning with {len(bins)-1} categories")
                
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in binning - {str(e)}")
            
        return df
    
    def _apply_pattern_groups(self, df, col, settings):
        """Apply pattern-based grouping"""
        patterns_str = settings.get("pattern_groups", "").strip()
        if not patterns_str:
            return df
            
        try:
            pattern_mapping = {}
            for item in patterns_str.split(","):
                if ":" in item:
                    pattern, label = item.split(":", 1)
                    pattern_mapping[pattern.strip()] = label.strip()
            
            if pattern_mapping:
                def apply_pattern_mapping(x):
                    x_str = str(x)
                    for pattern, label in pattern_mapping.items():
                        if re.search(pattern, x_str, re.IGNORECASE):
                            return label
                    return x_str
                
                original_unique = df[col].nunique()
                df[col] = df[col].apply(apply_pattern_mapping)
                new_unique = df[col].nunique()
                self._log_transformation(f"Column '{col}': Applied pattern grouping ({original_unique} → {new_unique} unique values)")
                
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in pattern grouping - {str(e)}")
            
        return df
    
    def _apply_encoding(self, df, col, settings):
        """Apply one-hot or label encoding"""
        if settings.get("onehot", False):
            try:
                # Check if column still exists (might have been removed by previous operations)
                if col not in df.columns:
                    return df
                    
                dummies = pd.get_dummies(df[col], prefix=col)
                if len(dummies.columns) > 1:  # Only if multiple categories
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    self._log_transformation(f"Column '{col}': Applied one-hot encoding ({len(dummies.columns)} new columns)")
                else:
                    # If only one category remains, use label encoding instead
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._log_transformation(f"Column '{col}': Applied label encoding (only one category remained)")
                    
            except Exception as e:
                self._log_transformation(f"Column '{col}': Error in one-hot encoding - {str(e)}")
                
        elif settings.get("label", False):
            try:
                # Check if column still exists
                if col not in df.columns:
                    return df
                    
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._log_transformation(f"Column '{col}': Applied label encoding")
            except Exception as e:
                self._log_transformation(f"Column '{col}': Error in label encoding - {str(e)}")
                
        return df
    
    def _apply_global_transformations(self, df, settings):
        """Apply global transformations"""
        
        # Global missing values removal
        if settings.get("remove_all_missing", False):
            original_rows = len(df)
            df = df.dropna()
            removed_rows = original_rows - len(df)
            if removed_rows > 0:
                self._log_transformation(f"Global: Removed {removed_rows} rows with any missing values")

        # --- 2. NEW: REMOVE RARE TARGET CLASSES ---
        if settings.get("remove_rare_target", False):
            target_col = settings.get("target_col")
            if target_col and target_col in df.columns:
                # Check if target is valid for this operation (not typically done on continuous floats)
                # But we do it if requested to be safe.
                try:
                    # Calculate counts
                    vc = df[target_col].value_counts()
                    # Find values appearing less than 2 times
                    rare_values = vc[vc < 2].index.tolist()
                    
                    if rare_values:
                        orig_len = len(df)
                        # Filter dataframe
                        df = df[~df[target_col].isin(rare_values)]
                        diff = orig_len - len(df)
                        self._log_transformation(f"Global: Removed {diff} rows with rare target values (count < 2): {rare_values}")
                except Exception as e:
                    self._log_transformation(f"Global: Failed to remove rare targets: {str(e)}")
        
        # Rest of your existing global transformations...
        global_enc = settings.get("encode_method", "none")
        if global_enc == "onehot":
            for col in df.select_dtypes(include=["object", "category"]).columns:
                try:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    if len(dummies.columns) > 1:
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        self._log_transformation(f"Global: One-hot encoded '{col}'")
                except:
                    pass
                    
        elif global_enc == "label":
            for col in df.select_dtypes(include=["object", "category"]).columns:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._log_transformation(f"Global: Label encoded '{col}'")
                except:
                    pass
        
        # Transpose
        if settings.get("transpose", False):
            df = df.T.reset_index()
            self._log_transformation("Applied transpose")
            
        # Replace negatives
        if settings.get("replace_negatives", False):
            rep_val = settings.get("replace_neg_value", "0")
            try:
                rep_val = float(rep_val)
            except:
                rep_val = 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    df[col] = df[col].apply(lambda x: rep_val if (isinstance(x, (int, float)) and x < 0) else x)
                    self._log_transformation(f"Replaced {negative_count} negative values in '{col}' with {rep_val}")
        
        return df
    
    def _log_transformation(self, message):
        """Add transformation to log"""
        self.transformations_log.append(message)
    
    def get_transformations_log(self):
        """Get all transformation messages"""
        return self.transformations_log
    
    def get_column_stats(self, column):
        """Get detailed statistics for a column"""
        if column not in self.df_original.columns:
            return None
            
        series = self.df_original[column]
        stats = {
            'dtype': str(series.dtype),
            'total_rows': len(series),
            'missing_values': series.isna().sum(),
            'missing_percentage': (series.isna().sum() / len(series)) * 100,
            'unique_values': series.nunique(),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'min': series.min(),
                'max': series.max(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std()
            })
        else:
            stats['top_values'] = series.value_counts().head(10).to_dict()
            
        return stats

    def _apply_clamping(self, df, col, settings):
        """Apply value clamping for numeric columns"""
        clamp_min = settings.get("clamp_min", "").strip()
        clamp_min_to = settings.get("clamp_min_to", "").strip()
        clamp_max = settings.get("clamp_max", "").strip()
        clamp_max_to = settings.get("clamp_max_to", "").strip()
        
        if not clamp_min and not clamp_max:
            return df
            
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            clamped_count = 0
            
            if clamp_min and clamp_min_to:
                try:
                    min_val = float(clamp_min)
                    min_to_val = float(clamp_min_to)
                    below_mask = numeric_series < min_val
                    clamped_count += below_mask.sum()
                    df.loc[below_mask, col] = min_to_val
                except ValueError:
                    pass
                    
            if clamp_max and clamp_max_to:
                try:
                    max_val = float(clamp_max)
                    max_to_val = float(clamp_max_to)
                    above_mask = numeric_series > max_val
                    clamped_count += above_mask.sum()
                    df.loc[above_mask, col] = max_to_val
                except ValueError:
                    pass
                    
            if clamped_count > 0:
                self._log_transformation(f"Column '{col}': Clamped {clamped_count} values")
                
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in clamping - {str(e)}")
            
        return df

    def _apply_percentage_rules(self, df, col, settings):
        """Apply percentage-based categorization"""
        percent_rules_str = settings.get("percent_rules", "").strip()
        if not percent_rules_str:
            return df
            
        try:
            # Parse rules like "10:High,40:Medium,50:Low"
            rules = []
            total_percent = 0
            for rule in percent_rules_str.split(","):
                if ":" in rule:
                    percent_str, label = rule.split(":", 1)
                    percent = float(percent_str.strip())
                    total_percent += percent
                    rules.append((percent, label.strip()))
                    
            if not rules or total_percent != 100:
                self._log_transformation(f"Column '{col}': Invalid percentage rules (total must be 100%)")
                return df
                
            # Sort values and apply percentage rules
            series = df[col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                sorted_series = series.sort_values(ascending=False)
            else:
                # For categorical, use frequency
                value_counts = series.value_counts()
                sorted_series = series.map(lambda x: (value_counts[x], x) if x in value_counts else (0, x))
                sorted_series = series.iloc[sorted_series.argsort()[::-1]]
                
            # Calculate cutoffs
            total_count = len(sorted_series)
            current_index = 0
            result_series = pd.Series(index=df.index, dtype=object)
            
            for percent, label in rules:
                count_for_label = int(total_count * percent / 100)
                end_index = min(current_index + count_for_label, total_count)
                
                if current_index < total_count:
                    indices_to_label = sorted_series.iloc[current_index:end_index].index
                    result_series.loc[indices_to_label] = label
                    current_index = end_index
                    
            # Fill any remaining with the last label
            if current_index < total_count:
                remaining_indices = sorted_series.iloc[current_index:].index
                result_series.loc[remaining_indices] = rules[-1][1] if rules else "Other"
                
            df[col] = result_series
            self._log_transformation(f"Column '{col}': Applied percentage-based categorization")
            
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in percentage rules - {str(e)}")
            
        return df
    
    def _apply_range_mapping(self, df, col, settings):
        """Apply range-based classification mapping"""
        range_mapping_str = settings.get("range_mapping", "").strip()
        if not range_mapping_str:
            return df
            
        try:
            # Parse range mappings like "3-9:A,10-20:B,21-30:C"
            range_mappings = []
            for mapping in range_mapping_str.split(","):
                if ":" in mapping:
                    range_part, label = mapping.split(":", 1)
                    range_part = range_part.strip()
                    label = label.strip()
                    
                    if "-" in range_part:
                        low, high = range_part.split("-")
                        range_mappings.append((float(low.strip()), float(high.strip()), label))
            
            if not range_mappings:
                return df
                
            def map_to_range(x):
                if pd.isna(x):
                    return x
                try:
                    x_val = float(x)
                    for low, high, label in range_mappings:
                        if low <= x_val <= high:
                            return label
                    return "Other"  # Value outside all ranges
                except (ValueError, TypeError):
                    return x  # Return original if not convertible to float
            
            # Apply mapping
            original_unique = df[col].nunique()
            df[col] = df[col].apply(map_to_range)
            new_unique = df[col].nunique()
            
            self._log_transformation(f"Column '{col}': Applied range mapping ({len(range_mappings)} ranges, {original_unique} → {new_unique} unique values)")
            
        except Exception as e:
            self._log_transformation(f"Column '{col}': Error in range mapping - {str(e)}")
            
        return df

# Keep the original Preprocessor for compatibility
class Preprocessor(AdvancedPreprocessor):
    """Backward compatibility"""
    pass