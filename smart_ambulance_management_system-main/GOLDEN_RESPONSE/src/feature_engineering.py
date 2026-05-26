"""
Feature Engineering Layer for the Intelligent Emergency Response Optimization System.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorized Haversine distance calculation in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in km
    return km

class FeatureEngineer:
    """
    Constructs advanced features from merged streams.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.features_cfg = config["features"]
        
        # Mappings and state to be computed from training data
        self.hospital_coords = {
            "City Hospital": (40.7128, -74.0060),
            "St. Jude Medical Center": (40.7589, -73.9851),
            "General Infirmary": (40.7829, -73.9654),
            "Mercy Hospital": (40.8068, -73.9568)
        }
        
        self.severity_map = {
            "Cardiac Arrest": 5,
            "Stroke": 4,
            "Trauma": 4,
            "Respiratory Distress": 3,
            "Minor Injury": 1
        }
        
        self.congestion_map = {
            "Low": 1.0,
            "Medium": 2.0,
            "High": 3.0,
            "Severe": 4.0
        }
        
        # Historical metrics computed from training data to avoid data leakage
        self.ambulance_efficiency = {}
        self.peak_hour_frequency = {}
        self.global_mean_duration = 15.0

    def fit_historical_metrics(self, df_train: pd.DataFrame) -> None:
        """
        Computes historical statistics on training data.
        """
        logger.info("Fitting historical efficiency metrics on training partition...")
        
        # 1. Historical ambulance response efficiency (mean response duration)
        self.ambulance_efficiency = df_train.groupby("ambulance_id")["response_duration_minutes"].mean().to_dict()
        self.global_mean_duration = df_train["response_duration_minutes"].mean()
        
        # 2. Peak-hour emergency frequency (count of emergencies by hour of day)
        df_train_hour = df_train.copy()
        df_train_hour["hour"] = df_train_hour["dispatch_time"].dt.hour
        hour_counts = df_train_hour.groupby("hour").size()
        # Normalize to get frequency (ratio of incidents in this hour vs total)
        self.peak_hour_frequency = (hour_counts / len(df_train_hour)).to_dict()

    def engineer_features(self, df_merged: pd.DataFrame, df_gps: pd.DataFrame, df_traffic: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Main entry point for feature engineering.
        """
        logger.info("Engineering predictive and operational features...")
        df = df_merged.copy()
        
        if is_train:
            self.fit_historical_metrics(df)
            
        # 1. Average congestion level during dispatch (numerical index)
        # Handle congestion level mapping
        congestion_col = "congestion_level"
        # Find matches from one-hot prefix if not a direct column
        if congestion_col in df.columns:
            df["congestion_index"] = df[congestion_col].map(self.congestion_map).fillna(1.0)
        else:
            # Reconstruct from one-hot columns if they exist
            df["congestion_index"] = 1.0
            for level, val in self.congestion_map.items():
                col_name = f"congestion_level_{level}"
                if col_name in df.columns:
                    df.loc[df[col_name] == 1, "congestion_index"] = val
                    
        # 2. Emergency severity index
        df["emergency_severity_index"] = df["emergency_type"].map(self.severity_map).fillna(1.0)
        
        # 3. Peak-hour emergency frequency
        df["hour_of_day"] = df["dispatch_time"].dt.hour
        df["peak_hour_frequency"] = df["hour_of_day"].map(self.peak_hour_frequency).fillna(0.04) # 1/24 fallback
        
        # 4. Historical ambulance response efficiency
        df["historical_ambulance_efficiency"] = df["ambulance_id"].map(self.ambulance_efficiency).fillna(self.global_mean_duration)
        
        # 5. Distance-to-incident (Straight line / Haversine)
        # Parse hospital coordinate mapping
        df["hospital_lat"] = df["hospital_destination"].map(lambda x: self.hospital_coords.get(x, (40.7128, -74.0060))[0])
        df["hospital_lon"] = df["hospital_destination"].map(lambda x: self.hospital_coords.get(x, (40.7128, -74.0060))[1])
        
        df["haversine_distance_km"] = haversine_np(
            df["hospital_lon"], df["hospital_lat"],
            df["incident_longitude"], df["incident_latitude"]
        )
        
        # 6. Distance-to-incident ratio (actual route distance from GPS vs straight line)
        # Calculate actual route distance for each dispatch from GPS
        gps_sorted = df_gps.sort_values(by=["ambulance_id", "timestamp"])
        gps_sorted["prev_lat"] = gps_sorted.groupby("ambulance_id")["latitude"].shift(1)
        gps_sorted["prev_lon"] = gps_sorted.groupby("ambulance_id")["longitude"].shift(1)
        
        gps_sorted["step_distance_km"] = haversine_np(
            gps_sorted["prev_lon"], gps_sorted["prev_lat"],
            gps_sorted["longitude"], gps_sorted["latitude"]
        ).fillna(0.0)
        
        logger.info("Computing spatial travel metrics from GPS path streams...")
        route_distances = []
        for idx, row in df.iterrows():
            amb_id = row["ambulance_id"]
            d_time = row["dispatch_time"]
            a_time = row["arrival_time"]
            
            mask = (
                (gps_sorted["ambulance_id"] == amb_id) & 
                (gps_sorted["timestamp"] >= d_time) & 
                (gps_sorted["timestamp"] <= a_time)
            )
            trip_gps = gps_sorted[mask]
            
            # Sum step distances
            actual_dist = trip_gps["step_distance_km"].sum()
            route_distances.append(actual_dist)
            
        df["actual_route_distance_km"] = route_distances
        # Ratio of actual vs straight line
        df["distance_to_incident_ratio"] = df["actual_route_distance_km"] / df["haversine_distance_km"].replace(0, 0.001)
        # Clean up bounds (ratio should be at least 1.0, fallback if 0)
        df["distance_to_incident_ratio"] = df["distance_to_incident_ratio"].clip(lower=1.0).fillna(1.2)
        
        # 7. Estimated travel delay
        # Normal speed assumed to be 50 km/h. Normal time = distance / 50 * 60 minutes
        df["normal_travel_time_minutes"] = (df["haversine_distance_km"] / 50.0) * 60.0
        df["estimated_travel_delay"] = df["response_duration_minutes"] - df["normal_travel_time_minutes"]
        # Delays cannot be negative realistically
        df["estimated_travel_delay"] = df["estimated_travel_delay"].clip(lower=0.0)
        
        # 8. Weather-adjusted travel speed
        # Speed scales down based on storm level and rain
        # speed = speed * (1.0 - 0.15 * storm_alert_level - 0.01 * rainfall)
        storm_val = df["storm_alert_level"].fillna(0.0)
        rain_val = df["rainfall"].fillna(0.0)
        speed_modifier = (1.0 - (0.15 * storm_val) - (0.01 * rain_val)).clip(lower=0.3)
        df["weather_adjusted_travel_speed"] = df["average_vehicle_speed"] * speed_modifier
        
        # 9. Route risk score
        # accident_reports is 0 or 1. road_closure_status can be closed.
        # Check if road_closure_status_Closed exists (if one-hot encoded)
        closure_factor = 0.0
        if "road_closure_status_Closed" in df.columns:
            closure_factor = df["road_closure_status_Closed"] * 3.0
        elif "road_closure_status" in df.columns:
            closure_factor = df["road_closure_status"].map(lambda x: 3.0 if x == "Closed" else 0.0).fillna(0.0)
            
        accident_factor = df["accident_reports"].fillna(0.0) * 2.0
        storm_factor = storm_val * 1.5
        
        df["route_risk_score"] = accident_factor + closure_factor + storm_factor + df["congestion_index"]
        
        # 10. Rolling traffic congestion trends (3-hour rolling average on segment)
        # Compute rolling congestion from the source traffic dataset to prevent lookahead bias
        logger.info("Computing rolling segment congestion averages...")
        traffic_df_sorted = df_traffic.copy()
        traffic_df_sorted["timestamp"] = pd.to_datetime(traffic_df_sorted["timestamp"])
        traffic_df_sorted = traffic_df_sorted.sort_values(by=["road_segment_id", "timestamp"])
        
        # Map congestion string to index in traffic DF
        traffic_df_sorted["congestion_index"] = traffic_df_sorted["congestion_level"].map(self.congestion_map).fillna(1.0)
        
        # Group by segment, compute rolling average of past 3 hours
        # Set time as index for rolling window computation
        traffic_df_sorted = traffic_df_sorted.set_index("timestamp")
        rolling_series = traffic_df_sorted.groupby("road_segment_id")["congestion_index"].rolling(
            window=self.features_cfg.get("rolling_congestion_window", "3h")
        ).mean().reset_index()
        
        rolling_series = rolling_series.rename(columns={"congestion_index": "rolling_congestion_avg"})
        
        # Join rolling average back to df on road_segment_id and dispatch_time_hour (rounded)
        df["dispatch_time_hour"] = df["dispatch_time"].dt.round("h")
        df = pd.merge(
            df,
            rolling_series,
            left_on=["dispatch_time_hour", "road_segment_id"],
            right_on=["timestamp", "road_segment_id"],
            how="left"
        )
        df = df.drop(columns=["dispatch_time_hour", "timestamp"], errors="ignore")
        # Fill missing rolling averages with standard congestion index
        df["rolling_congestion_avg"] = df["rolling_congestion_avg"].fillna(df["congestion_index"])
        
        # 11. Nearby hospital load indicators
        # Count of active runs to same hospital at the time of dispatch (+/- 15 minutes window)
        logger.info("Computing hospital load indicators...")
        hospital_loads = []
        for idx, row in df.iterrows():
            hosp = row["hospital_destination"]
            d_time = row["dispatch_time"]
            
            # Active runs are those dispatched before this dispatch, and arrived after this dispatch,
            # or dispatched within 30 minutes of this dispatch
            active_mask = (
                (df["hospital_destination"] == hosp) & 
                (df["dispatch_time"] <= d_time) & 
                (df["arrival_time"] >= d_time)
            )
            hospital_loads.append(active_mask.sum() - 1) # exclude self
            
        df["nearby_hospital_load_indicator"] = hospital_loads
        # Cap load at 0
        df["nearby_hospital_load_indicator"] = df["nearby_hospital_load_indicator"].clip(lower=0)
        
        # Drop temporary coordinate columns
        df = df.drop(columns=["hospital_lat", "hospital_lon"], errors="ignore")
        
        logger.info("Feature engineering completed.")
        return df
