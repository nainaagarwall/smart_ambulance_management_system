"""
Pipeline Orchestrator for the Intelligent Emergency Response Optimization System.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.config import load_config
from src.data_generator import generate_synthetic_data
from src.ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelManager
from src.visualization import ResponseVisualizer
from src.reporting import ResponseReporter

logger = logging.getLogger(__name__)

class EmergencyResponsePipeline:
    """
    Orchestrates the entire data preparation, modeling, visualization,
    and reporting pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Initialize modules
        self.ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_manager = ModelManager(self.config)
        self.visualizer = ResponseVisualizer(self.config)
        self.reporter = ResponseReporter(self.config)

    def run_pipeline(self, generate_data: bool = True) -> Tuple[Dict[str, Any], str]:
        """
        Executes the end-to-end pipeline:
        1. Checks/Generates raw data
        2. Ingests raw data
        3. Cleans, interpolates GPS, repairs timestamps, filters outliers
        4. Aligns and merges datasets spatially and temporally
        5. Performs category encoding and scale normalization
        6. Computes 10 engineered features
        7. Prepares target fields and trains/selects best models
        8. Evaluates models and saves them (joblib)
        9. Exports alerts (JSON) and summaries (CSV)
        10. Generates 5 analytics visualizations (PNG)
        """
        logger.info("=========================================")
        logger.info("STARTING INTELLIGENT EMERGENCY RESPONSE SYSTEM")
        logger.info("=========================================")
        
        # Step 1: Data Check / Generation
        raw_dir = self.config["data"]["raw_dir"]
        dispatch_path = self.config["data"]["dispatch_file"]
        
        if generate_data or not os.path.exists(dispatch_path):
            logger.info("Generating synthetic sample datasets...")
            generate_synthetic_data(raw_dir, seed=self.config["system"]["random_seed"])
            
        # Step 2: Ingestion
        logger.info("[Step 2/8] Ingesting raw datasets...")
        df_dispatch, df_gps, df_traffic, df_weather = self.ingestion.ingest_all(self.config)
        
        # Step 3: Cleaning & Preprocessing
        logger.info("[Step 3/8] Cleaning and preprocessing raw streams...")
        # Repair timestamps in dispatch
        df_dispatch_repaired = self.preprocessor.repair_timestamps(df_dispatch)
        
        # Interpolate missing GPS coordinates
        df_gps_interpolated = self.preprocessor.interpolate_gps_coordinates(df_gps)
        
        # Outlier filtering
        df_dispatch_clean, df_gps_clean = self.preprocessor.remove_outliers(
            df_dispatch_repaired, df_gps_interpolated
        )
        
        # Step 4: Spatial-Temporal Merging
        logger.info("[Step 4/8] Merging datasets via spatial and temporal keys...")
        df_merged = self.preprocessor.align_and_merge_datasets(
            df_dispatch_clean, df_gps_clean, df_traffic, df_weather
        )
        
        # Step 5: Feature Engineering
        logger.info("[Step 5/8] Performing feature engineering...")
        df_features = self.feature_engineer.engineer_features(
            df_merged, df_gps_clean, df_traffic, is_train=True
        )
        
        # Normalize and encode
        df_encoded, cols_scaled = self.preprocessor.encode_and_normalize(df_features, is_train=True)
        
        # Step 6: Model Training & Auto-Selection
        logger.info("[Step 6/8] Training and evaluating ML models...")
        eval_report, best_models = self.model_manager.train_and_select_best(df_encoded)
        
        # Predict delay probability for the entire dataset using the best "is_delayed" model
        logger.info("[Step 7/8] Generating predictions for reporting...")
        best_delay_model_pkg = best_models["is_delayed"]
        best_delay_model = best_delay_model_pkg["model_name"]
        
        # Re-prepare target dataframe to get label and feature columns align
        df_prepared = self.model_manager.prepare_targets(df_encoded)
        feature_cols = best_delay_model_pkg["features"]
        
        X_all = df_prepared[feature_cols]
        # Predict probabilities
        model_object = joblib.load(best_delay_model_pkg["save_path"])["model"]
        y_prob_all = model_object.predict_proba(X_all)[:, 1]
        
        # Append delay prediction and actual predictions to dataframe for reporting
        df_prepared["predicted_delay_probability"] = y_prob_all
        
        # Step 7: Export Reports
        alert_path = self.reporter.generate_dispatch_alerts(df_prepared, y_prob_all)
        admin_path = self.reporter.generate_admin_summary(df_prepared)
        
        # Step 8: Visualization
        logger.info("[Step 8/8] Generating analytics plots...")
        self.visualizer.generate_all_plots(df_dispatch_clean, df_gps_clean, df_traffic, y_prob_all)
        
        # Create a text summary of the evaluations
        summary_md_path = self.generate_text_summary(eval_report, best_models)
        
        logger.info("=========================================")
        logger.info("PIPELINE EXECUTED SUCCESSFULLY")
        logger.info("=========================================")
        
        return eval_report, summary_md_path

    def generate_text_summary(self, eval_report: Dict[str, Any], best_models: Dict[str, Any]) -> str:
        """
        Generates a markdown text evaluation summary.
        """
        output_dir = self.config["data"].get("output_dir", "outputs")
        summary_path = os.path.join(output_dir, "model_evaluation_report.md")
        
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        lines.append("# Model Evaluation Report: Intelligent Emergency Response Optimization")
        lines.append("")
        lines.append("This report summarizes the performance evaluation and model comparisons for the emergency response predictive models.")
        lines.append("")
        lines.append("## Target Definitions & Selections")
        lines.append("")
        lines.append("| Target Variable | Type | Best Model Selected | Test Accuracy | F1-Score |")
        lines.append("|---|---|---|---|---|")
        
        for target, pkg in best_models.items():
            t_type = "Binary Classification" if target == "is_delayed" else "Multi-class Classification"
            lines.append(f"| `{target}` | {t_type} | **{pkg['model_name']}** | {pkg['metrics']['accuracy']:.4f} | {pkg['metrics']['f1']:.4f} |")
            
        lines.append("")
        lines.append("## Detailed Performance Comparison")
        lines.append("")
        
        for target, models in eval_report.items():
            lines.append(f"### Target: `{target}`")
            lines.append("")
            lines.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
            lines.append("|---|---|---|---|---|---|")
            
            for m_name, metrics in models.items():
                lines.append(f"| {m_name.replace('_', ' ').capitalize()} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |")
            
            lines.append("")
            lines.append("#### Confusion Matrices")
            lines.append("")
            for m_name, metrics in models.items():
                lines.append(f"**{m_name.replace('_', ' ').capitalize()} Confusion Matrix:**")
                lines.append("```")
                lines.append(str(metrics["confusion_matrix"]))
                lines.append("```")
                lines.append("")
                
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
            
        logger.info("Saved evaluation summary markdown report to %s", summary_path)
        return summary_path
