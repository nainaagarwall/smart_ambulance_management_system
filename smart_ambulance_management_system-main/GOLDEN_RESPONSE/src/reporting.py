"""
Reporting and Export Layer for the Intelligent Emergency Response Optimization System.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ResponseReporter:
    """
    Handles generation of JSON dispatcher alerts and CSV administrator reports.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.reports_dir = config["data"].get("reports_dir", "outputs/reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_dispatch_alerts(self, df_merged: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.70) -> str:
        """
        Generates real-time delayed-response risk alerts in JSON format for the dispatch center.
        """
        logger.info("Generating real-time delayed-response risk alerts...")
        
        alerts = []
        for idx, row in df_merged.iterrows():
            prob = y_prob[idx]
            
            # If the predicted probability of delay exceeds the threshold (e.g. 70%)
            if prob >= threshold:
                incident_id = row.get("incident_id", "INC_UNKNOWN")
                amb_id = row.get("ambulance_id", "AMB_UNKNOWN")
                etype = row.get("emergency_type", "Unknown")
                hosp = row.get("hospital_destination", "Unknown")
                severity = int(row.get("emergency_severity_index", 1))
                risk_score = float(row.get("route_risk_score", 1.0))
                congestion = float(row.get("congestion_index", 1.0))
                
                # Determine recommended actions based on factors
                if severity >= 4 and congestion >= 3.0:
                    recommended_action = "IMMEDIATE ESCALATION: Re-route ambulance via green segment or request police escort due to high traffic."
                elif congestion >= 3.0:
                    recommended_action = "ADVISORY: Re-route ambulance to clear segment. Traffic is heavy."
                elif risk_score >= 4.0:
                    recommended_action = "CAUTION: Heavy weather/accidents reported on segment. Request driver exercise extreme caution."
                else:
                    recommended_action = "STANDBY: Dispatch secondary backup unit from neighboring zone in case of further delays."
                    
                alerts.append({
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "incident_id": incident_id,
                    "ambulance_id": amb_id,
                    "emergency_type": etype,
                    "hospital_destination": hosp,
                    "predicted_delay_probability": round(float(prob), 3),
                    "emergency_severity_index": severity,
                    "route_risk_score": round(risk_score, 2),
                    "recommended_action": recommended_action
                })
                
        # Save as JSON
        save_path = os.path.join(self.reports_dir, "delayed_response_alerts.json")
        with open(save_path, "w") as f:
            json.dump(alerts, f, indent=4)
            
        logger.info("Saved %d dispatch alerts to %s", len(alerts), save_path)
        return save_path

    def generate_admin_summary(self, df_merged_with_predictions: pd.DataFrame) -> str:
        """
        Generates CSV analytics summaries by geographic zone for city administrators.
        Summarizes incident frequencies, average delays, and suggests optimal deployment numbers.
        """
        logger.info("Generating city administrator analytics summaries...")
        
        df = df_merged_with_predictions.copy()
        
        # Group by optimal_zone
        zone_summary = []
        
        # If optimal_zone is not present, create it
        if "optimal_zone" not in df.columns:
            # Recreate zone
            grid_size = self.config["features"].get("spatial_grid_size_km", 1.0) / 111.0
            def get_grid_zone(lat, lon):
                if pd.isna(lat) or pd.isna(lon):
                    return "Zone_Unknown"
                return f"Zone_{int(lat/grid_size)}_{int(lon/grid_size)}"
            df["optimal_zone"] = df.apply(lambda r: get_grid_zone(r["incident_latitude"], r["incident_longitude"]), axis=1)
            
        grouped = df.groupby("optimal_zone")
        
        for zone, group in grouped:
            total_emergencies = len(group)
            avg_duration = group["response_duration_minutes"].mean()
            
            # Delayed ratio
            threshold = self.config["features"].get("delayed_response_threshold_minutes", 15.0)
            delayed_count = (group["response_duration_minutes"] > threshold).sum()
            delayed_ratio = delayed_count / total_emergencies if total_emergencies > 0 else 0.0
            
            avg_congestion = group["congestion_index"].mean() if "congestion_index" in group.columns else 1.0
            avg_risk = group["route_risk_score"].mean() if "route_risk_score" in group.columns else 1.0
            avg_hospital_load = group["nearby_hospital_load_indicator"].mean() if "nearby_hospital_load_indicator" in group.columns else 0.0
            
            # Suggest recommended deployment count (Simple heuristic: base of 1, plus additional units for high frequency/delays/congestion)
            recommended_deployment = 1
            if total_emergencies > 100:
                recommended_deployment += 2
            elif total_emergencies > 30:
                recommended_deployment += 1
                
            if delayed_ratio > 0.3:
                recommended_deployment += 1
            if avg_congestion > 2.5:
                recommended_deployment += 1
                
            zone_summary.append({
                "zone_id": zone,
                "total_emergencies": total_emergencies,
                "average_response_duration_minutes": round(avg_duration, 1),
                "delayed_ratio_percent": round(delayed_ratio * 100, 1),
                "average_congestion_index": round(avg_congestion, 2),
                "average_route_risk_score": round(avg_risk, 2),
                "average_nearby_hospital_load": round(avg_hospital_load, 1),
                "recommended_standby_ambulances": recommended_deployment
            })
            
        df_summary = pd.DataFrame(zone_summary)
        # Sort by total emergencies desc
        df_summary = df_summary.sort_values(by="total_emergencies", ascending=False).reset_index(drop=True)
        
        # Save as CSV
        save_path = os.path.join(self.reports_dir, "city_admin_analytics_summary.csv")
        df_summary.to_csv(save_path, index=False)
        
        logger.info("Saved city admin analytics summary to %s", save_path)
        return save_path
