"""
Machine Learning Modeling Layer for the Intelligent Emergency Response Optimization System.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages model definition, training, evaluation, comparison, and serialization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models_cfg = config["models"]
        self.seed = config["system"]["random_seed"]
        self.saved_models = {}

    def get_grid_zone(self, lat: float, lon: float, grid_size: float = 0.05) -> str:
        """
        Partitions the coordinate space into categorical grid zones.
        Uses a simple spatial grid naming system.
        """
        if pd.isna(lat) or pd.isna(lon):
            return "Zone_Unknown"
        lat_idx = int(lat / grid_size)
        lon_idx = int(lon / grid_size)
        return f"Zone_{lat_idx}_{lon_idx}"

    def prepare_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates target variables:
        1. is_delayed (Binary Classification): 1 if response duration > threshold, else 0
        2. optimal_zone (Multi-class Classification): Grid zone name where emergency occurred
        3. arrival_category (Multi-class Classification): response duration range:
           - 0: < 10 mins (Fast)
           - 1: 10-20 mins (Standard)
           - 2: > 20 mins (Delayed)
        """
        df_target = df.copy()
        
        # 1. is_delayed
        threshold = self.config["features"].get("delayed_response_threshold_minutes", 15.0)
        df_target["is_delayed"] = (df_target["response_duration_minutes"] > threshold).astype(int)
        
        # 2. optimal_zone (stationing deployment zone)
        grid_size = self.config["features"].get("spatial_grid_size_km", 1.0) / 111.0 # degrees roughly
        df_target["optimal_zone"] = df_target.apply(
            lambda r: self.get_grid_zone(r["incident_latitude"], r["incident_longitude"], grid_size), 
            axis=1
        )
        
        # 3. arrival_category
        def categorize_arrival(duration):
            if duration < 10.0:
                return 0 # Fast
            elif duration <= 20.0:
                return 1 # Standard
            else:
                return 2 # Delayed
                
        df_target["arrival_category"] = df_target["response_duration_minutes"].apply(categorize_arrival)
        
        return df_target

    def get_features_list(self, df: pd.DataFrame) -> List[str]:
        """
        Selects all numerical, normalized, and encoded features to feed into the model.
        Excludes raw IDs, string locations, timestamps, and target columns.
        """
        exclude_cols = [
            "incident_id", "ambulance_id", "dispatch_time", "arrival_time", 
            "incident_location", "road_segment_id", "weather_timestamp",
            "incident_latitude", "incident_longitude",
            "response_duration_minutes", "is_delayed", "optimal_zone", "arrival_category",
            "normal_travel_time_minutes"
        ]
        
        # We also exclude non-normalized continuous columns if we have their normalized versions
        continuous_raw = ["rainfall", "visibility", "temperature", "average_vehicle_speed", "actual_gps_speed"]
        for col in continuous_raw:
            if f"{col}_normalized" in df.columns:
                exclude_cols.append(col)
                
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_cols.append(col)
        return feature_cols

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, is_multiclass: bool = False) -> Dict[str, Any]:
        """
        Computes evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Precision, Recall, F1
        if is_multiclass:
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            # ROC-AUC multi-class
            try:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except Exception:
                roc_auc = 0.5
        else:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            # ROC-AUC binary (use probability of positive class)
            try:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            except Exception:
                roc_auc = 0.5
                
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }

    def train_and_select_best(self, df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Trains and compares Random Forest and Gradient Boosting Classifiers for each of the three targets.
        Saves the best-performing model for each target.
        """
        df_prepared = self.prepare_targets(df)
        feature_cols = self.get_features_list(df_prepared)
        
        logger.info("Training features list (%d features): %s", len(feature_cols), feature_cols)
        
        targets = {
            "is_delayed": {"multiclass": False},
            "optimal_zone": {"multiclass": True},
            "arrival_category": {"multiclass": True}
        }
        
        evaluation_report = {}
        best_models = {}
        
        # Ensure outputs/models dir exists
        models_dir = self.config["data"].get("models_dir", "outputs/models")
        os.makedirs(models_dir, exist_ok=True)
        
        for target_name, properties in targets.items():
            is_multi = properties["multiclass"]
            logger.info("--- Training Models for Target: %s ---", target_name)
            
            # Filter out records where target or features might be null (pre-cleaned, but safety check)
            df_target_clean = df_prepared.dropna(subset=[target_name] + feature_cols)
            
            X = df_target_clean[feature_cols]
            y = df_target_clean[target_name]
            
            # Map target labels to sequential integers if they are string categories (e.g. for optimal_zone)
            label_mapping = None
            if is_multi and y.dtype == object:
                unique_labels = sorted(y.unique())
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                y = y.map(label_mapping)
                logger.info("Mapped labels for target %s: %s", target_name, label_mapping)
                
            # Check if stratification is possible (all classes must have at least 2 members)
            can_stratify = False
            if len(y.unique()) > 1:
                class_counts = y.value_counts()
                if class_counts.min() >= 2:
                    can_stratify = True
                    
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.models_cfg.get("test_size", 0.2), 
                random_state=self.seed if hasattr(self, "seed") else 42,
                stratify=y if can_stratify else None
            )
            
            # 1. Random Forest Classifier
            rf_params = self.models_cfg.get("random_forest", {})
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train, y_train)
            rf_eval = self.evaluate_model(rf_model, X_test, y_test, is_multiclass=is_multi)
            
            # 2. Gradient Boosting Classifier
            gb_params = self.models_cfg.get("gradient_boosting", {})
            gb_model = GradientBoostingClassifier(**gb_params)
            gb_model.fit(X_train, y_train)
            gb_eval = self.evaluate_model(gb_model, X_test, y_test, is_multiclass=is_multi)
            
            evaluation_report[target_name] = {
                "random_forest": rf_eval,
                "gradient_boosting": gb_eval
            }
            
            # Compare and select best (based on Accuracy, fallback to F1)
            rf_score = rf_eval["accuracy"]
            gb_score = gb_eval["accuracy"]
            
            if rf_score >= gb_score:
                best_model = rf_model
                best_name = "random_forest"
                best_eval = rf_eval
            else:
                best_model = gb_model
                best_name = "gradient_boosting"
                best_eval = gb_eval
                
            logger.info("Selected %s as the best model for target %s (Accuracy: %.4f vs %.4f).", 
                        best_name, target_name, best_eval["accuracy"], min(rf_score, gb_score))
            
            # Save metadata and actual model object
            model_save_path = os.path.join(models_dir, f"{target_name}_best_model.joblib")
            
            # Bundle label mapping inside saved dict if multi-class
            save_payload = {
                "model": best_model,
                "features": feature_cols,
                "label_mapping": label_mapping,
                "accuracy": best_eval["accuracy"],
                "f1": best_eval["f1"]
            }
            
            joblib.dump(save_payload, model_save_path)
            logger.info("Saved best model to %s", model_save_path)
            
            best_models[target_name] = {
                "model_name": best_name,
                "metrics": best_eval,
                "features": feature_cols,
                "save_path": model_save_path,
                "label_mapping": label_mapping
            }
            
        self.saved_models = best_models
        return evaluation_report, best_models
