"""
Data Preprocessing Layer for the Intelligent Emergency Response Optimization System.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Cleans, repairs, interpolates, filters outliers, aligns, and normalizes
    the emergency dispatch and geospatial tracking streams.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessing_cfg = config["preprocessing"]
        self.scaler = StandardScaler()
        self.fitted_features = []

    def repair_timestamps(self, df_dispatch: pd.DataFrame) -> pd.DataFrame:
        """
        Detects and repairs corrupted timestamps.
        - Parses times to datetime objects.
        - Fixes cases where arrival_time < dispatch_time.
        - Imputes missing arrival times using the median response duration.
        """
        df = df_dispatch.copy()
        
        # Parse to datetime
        df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
        df["arrival_time"] = pd.to_datetime(df["arrival_time"])
        
        # Calculate initial response duration
        df["response_duration_minutes"] = pd.to_numeric(df["response_duration_minutes"], errors='coerce')
        
        # Identify bad records: negative duration, arrival before dispatch, or missing duration
        is_negative = df["response_duration_minutes"] <= 0
        is_inverted = df["arrival_time"] < df["dispatch_time"]
        is_missing = df["arrival_time"].isna() | df["response_duration_minutes"].isna()
        
        corrupted_mask = is_negative | is_inverted | is_missing
        corrupted_count = corrupted_mask.sum()
        
        if corrupted_count > 0:
            logger.info("Detecting and repairing %d corrupted timestamps in dispatch records.", corrupted_count)
            
            # Calculate median response duration per emergency type as fallback
            medians = df[~corrupted_mask].groupby("emergency_type")["response_duration_minutes"].median().to_dict()
            global_median = df[~corrupted_mask]["response_duration_minutes"].median()
            if pd.isna(global_median) or global_median <= 0:
                global_median = self.preprocessing_cfg.get("corrupted_time_fallback_minutes", 15.0)
                
            # Impute response duration
            def get_median_duration(row):
                if corrupted_mask.loc[row.name]:
                    etype = row["emergency_type"]
                    return medians.get(etype, global_median)
                return row["response_duration_minutes"]
                
            df["response_duration_minutes"] = df.apply(get_median_duration, axis=1)
            
            # Recompute arrival_time for repaired rows
            repaired_arrival = df["dispatch_time"] + pd.to_timedelta(df["response_duration_minutes"], unit='m')
            df.loc[corrupted_mask, "arrival_time"] = repaired_arrival.loc[corrupted_mask]
            
        return df

    def interpolate_gps_coordinates(self, df_gps: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts GPS data and interpolates missing coordinates (NaNs) linearly 
        grouped by ambulance.
        """
        df = df_gps.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["ambulance_id", "timestamp"]).reset_index(drop=True)
        
        initial_missing_lat = df["latitude"].isna().sum()
        initial_missing_lon = df["longitude"].isna().sum()
        
        if initial_missing_lat > 0 or initial_missing_lon > 0:
            logger.info("Interpolating missing GPS coordinates: %d missing latitudes, %d missing longitudes.", 
                        initial_missing_lat, initial_missing_lon)
            
            # Group by ambulance and interpolate
            limit = self.preprocessing_cfg.get("gps_interpolation_limit", 5)
            
            # Note: We interpolate latitude and longitude columns
            df["latitude"] = df.groupby("ambulance_id", group_keys=False)["latitude"].apply(
                lambda x: x.interpolate(method="linear", limit=limit)
            )
            df["longitude"] = df.groupby("ambulance_id", group_keys=False)["longitude"].apply(
                lambda x: x.interpolate(method="linear", limit=limit)
            )
            
            # Drop remaining rows that could not be interpolated
            still_missing = df["latitude"].isna() | df["longitude"].isna()
            missing_count = still_missing.sum()
            if missing_count > 0:
                df = df[~still_missing].reset_index(drop=True)
                logger.warning("Dropped %d GPS records that couldn't be interpolated within limit.", missing_count)
                
        return df

    def remove_outliers(self, df_dispatch: pd.DataFrame, df_gps: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters out outlier values:
        - Dispatch: response duration not within configured limits.
        - GPS: travel speed not within limits.
        """
        # 1. Clean dispatch response duration outliers
        d_min = self.preprocessing_cfg.get("min_allowable_speed", 1.0) # wait, response duration is in minutes
        d_max = self.preprocessing_cfg.get("max_response_duration", 120.0)
        
        initial_dispatch_len = len(df_dispatch)
        df_dispatch_clean = df_dispatch[
            (df_dispatch["response_duration_minutes"] >= 1.0) & 
            (df_dispatch["response_duration_minutes"] <= d_max)
        ].copy()
        
        removed_dispatch = initial_dispatch_len - len(df_dispatch_clean)
        if removed_dispatch > 0:
            logger.info("Removed %d response duration outliers from dispatch records.", removed_dispatch)
            
        # 2. Clean GPS travel speed outliers
        s_max = self.preprocessing_cfg.get("max_allowable_speed", 150.0)
        s_min = self.preprocessing_cfg.get("min_allowable_speed", 2.0)
        
        initial_gps_len = len(df_gps)
        # We allow speeds down to 0, but anything above s_max is filtered
        df_gps_clean = df_gps[
            (df_gps["current_speed"] <= s_max) & 
            ((df_gps["current_speed"] >= s_min) | (df_gps["current_speed"] == 0.0))
        ].copy()
        
        removed_gps = initial_gps_len - len(df_gps_clean)
        if removed_gps > 0:
            logger.info("Removed %d travel speed outliers from GPS tracking.", removed_gps)
            
        return df_dispatch_clean, df_gps_clean

    def align_and_merge_datasets(
        self, 
        df_dispatch: pd.DataFrame, 
        df_gps: pd.DataFrame, 
        df_traffic: pd.DataFrame, 
        df_weather: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges datasets using aligned spatial and temporal keys.
        - Aligns Incident Location with the nearest Road Segment.
        - Merges Weather data using hourly aligned dispatch timestamps.
        - Merges Traffic data using hourly aligned dispatch timestamps and the nearest Road Segment.
        - Integrates average actual travel speed of the ambulance during that dispatch period from GPS records.
        """
        logger.info("Aligning and merging datasets using spatial and temporal keys...")
        df = df_dispatch.copy()
        
        # 1. Parse incident_location to Latitude and Longitude
        df[["incident_latitude", "incident_longitude"]] = df["incident_location"].str.split(",", expand=True).astype(float)
        
        # 2. Extract road segments locations from GPS tracking to perform spatial alignment
        # We find the mean lat/lon for each segment in the GPS data
        segment_coords = df_gps.dropna(subset=["latitude", "longitude"]).groupby("route_segment_id")[["latitude", "longitude"]].mean().reset_index()
        
        # Map each incident to the closest road segment (Spatial alignment)
        logger.info("Performing spatial mapping of incident locations to nearest road segments...")
        def find_closest_segment(row):
            lat, lon = row["incident_latitude"], row["incident_longitude"]
            # Fast vectorized distance calculation (Euclidean approximation is sufficient for this mapping)
            dists = np.sqrt((segment_coords["latitude"] - lat)**2 + (segment_coords["longitude"] - lon)**2)
            return segment_coords.loc[dists.argmin(), "route_segment_id"]
            
        df["road_segment_id"] = df.apply(find_closest_segment, axis=1)
        
        # 3. Temporal alignment keys (hourly rounding)
        df["dispatch_time_hour"] = df["dispatch_time"].dt.round("h")
        
        df_weather = df_weather.copy()
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
        df_weather["weather_hour"] = df_weather["timestamp"].dt.round("h")
        # Keep only unique hours for clean join
        df_weather = df_weather.drop_duplicates(subset=["weather_hour"])
        
        df_traffic = df_traffic.copy()
        df_traffic["timestamp"] = pd.to_datetime(df_traffic["timestamp"])
        df_traffic["traffic_hour"] = df_traffic["timestamp"].dt.round("h")
        # Keep unique combinations of traffic hour and segment
        df_traffic = df_traffic.drop_duplicates(subset=["traffic_hour", "road_segment_id"])
        
        # 4. Merging Weather (on dispatch hour)
        df_merged = pd.merge(
            df, 
            df_weather, 
            left_on="dispatch_time_hour", 
            right_on="weather_hour", 
            how="left"
        )
        df_merged = df_merged.drop(columns=["weather_hour", "timestamp_y"], errors="ignore")
        df_merged = df_merged.rename(columns={"timestamp_x": "weather_timestamp"})
        
        # 5. Merging Traffic (on dispatch hour and segment ID)
        df_merged = pd.merge(
            df_merged, 
            df_traffic, 
            left_on=["dispatch_time_hour", "road_segment_id"], 
            right_on=["traffic_hour", "road_segment_id"], 
            how="left"
        )
        df_merged = df_merged.drop(columns=["traffic_hour", "timestamp"], errors="ignore")
        
        # 6. Extract average travel speed and details from GPS records for this run (Temporal window join)
        logger.info("Computing actual trip metrics from GPS streams...")
        gps_speed_list = []
        for idx, row in df_merged.iterrows():
            amb_id = row["ambulance_id"]
            d_time = row["dispatch_time"]
            a_time = row["arrival_time"]
            
            # Find GPS reports for this ambulance during the response window
            mask = (
                (df_gps["ambulance_id"] == amb_id) & 
                (df_gps["timestamp"] >= d_time) & 
                (df_gps["timestamp"] <= a_time)
            )
            trip_gps = df_gps[mask]
            
            if not trip_gps.empty:
                avg_gps_speed = trip_gps["current_speed"].mean()
            else:
                # Fallback to average segment speed from traffic monitoring
                avg_gps_speed = row["average_vehicle_speed"] if not pd.isna(row["average_vehicle_speed"]) else 40.0
                
            gps_speed_list.append(avg_gps_speed)
            
        df_merged["actual_gps_speed"] = gps_speed_list
        
        # Cleanup temp columns
        df_merged = df_merged.drop(columns=["dispatch_time_hour"], errors="ignore")
        
        logger.info("Merged dataset successfully created: %d rows", len(df_merged))
        return df_merged

    def encode_and_normalize(self, df_merged: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        One-hot encodes categorical fields and scales/normalizes continuous features.
        """
        df = df_merged.copy()
        
        # Categorical columns to encode
        cat_cols = ["emergency_type", "hospital_destination", "congestion_level", "road_closure_status"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                
        # Perform One-hot Encoding
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        
        # Identify continuous features to normalize
        continuous_cols = [
            "rainfall", "visibility", "temperature", "average_vehicle_speed", 
            "actual_gps_speed"
        ]
        
        # Filter existing columns
        cols_to_scale = [col for col in continuous_cols if col in df.columns]
        
        for col in cols_to_scale:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)
            
        if cols_to_scale:
            if is_train:
                self.scaler.fit(df[cols_to_scale])
                self.fitted_features = cols_to_scale
            
            scaled_data = self.scaler.transform(df[cols_to_scale])
            for i, col in enumerate(cols_to_scale):
                df[f"{col}_normalized"] = scaled_data[:, i]
                
        # Fill any other numeric NAs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df, cols_to_scale
