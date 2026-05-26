"""
Data Ingestion Layer for the Intelligent Emergency Response Optimization System.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles loading, basic schema validation, and cleansing of the raw CSV files.
    """
    
    # Expected schemas for raw files
    SCHEMAS = {
        "dispatch": [
            "incident_id", "ambulance_id", "dispatch_time", "arrival_time", 
            "incident_location", "hospital_destination", "emergency_type", 
            "response_duration_minutes"
        ],
        "gps": [
            "ambulance_id", "timestamp", "latitude", "longitude", "current_speed", "route_segment_id"
        ],
        "traffic": [
            "timestamp", "road_segment_id", "congestion_level", "average_vehicle_speed", 
            "accident_reports", "road_closure_status"
        ],
        "weather": [
            "timestamp", "rainfall", "visibility", "temperature", "storm_alert_level"
        ]
    }

    @staticmethod
    def load_csv(file_path: str, dataset_name: str) -> pd.DataFrame:
        """
        Loads a CSV file and validates its basic structure.
        """
        if not os.path.exists(file_path):
            logger.error("%s file not found at path: %s", dataset_name.capitalize(), file_path)
            raise FileNotFoundError(f"{dataset_name.capitalize()} file not found at: {file_path}")
            
        try:
            df = pd.read_csv(file_path)
            logger.info("Successfully loaded %s dataset: %d rows, %d columns.", 
                        dataset_name, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.critical("Failed to read CSV at %s: %s", file_path, str(e))
            raise ValueError(f"Corrupted or invalid CSV file {file_path}: {e}")

    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Checks if required columns are present in the DataFrame.
        """
        expected_cols = self.SCHEMAS.get(dataset_name)
        if not expected_cols:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.error("Schema validation failed for %s. Missing columns: %s", dataset_name, missing_cols)
            raise ValueError(f"Schema validation failed for {dataset_name}. Missing columns: {missing_cols}")
        
        logger.debug("Schema validation passed for %s.", dataset_name)

    def clean_duplicates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Removes duplicate incident entries and duplicate logs.
        """
        initial_count = len(df)
        if dataset_name == "dispatch":
            # Remove duplicate incidents
            df = df.drop_duplicates(subset=["incident_id"], keep="first")
        elif dataset_name == "gps":
            # Remove exact duplicate tracks
            df = df.drop_duplicates(subset=["ambulance_id", "timestamp"], keep="first")
        elif dataset_name == "traffic":
            # Remove duplicate segment logs at same time
            df = df.drop_duplicates(subset=["timestamp", "road_segment_id"], keep="first")
        elif dataset_name == "weather":
            df = df.drop_duplicates(subset=["timestamp"], keep="first")
            
        final_count = len(df)
        removed = initial_count - final_count
        if removed > 0:
            logger.info("Removed %d duplicate rows from %s dataset.", removed, dataset_name)
        return df

    def validate_coordinates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Validates latitude and longitude values. Coordinates that are physically 
        out of range (Lat: -90 to 90, Lon: -180 to 180) are treated as invalid and dropped.
        Note: NaN values are allowed at this step as they are handled in preprocessing.
        """
        lat_col, lon_col = None, None
        if "latitude" in df.columns and "longitude" in df.columns:
            lat_col, lon_col = "latitude", "longitude"
            
        if not lat_col:
            return df
            
        initial_len = len(df)
        
        # Valid range masks (ignoring NaNs)
        valid_lat = (df[lat_col].isna()) | (df[lat_col] >= -90.0) & (df[lat_col] <= 90.0)
        valid_lon = (df[lon_col].isna()) | (df[lon_col] >= -180.0) & (df[lon_col] <= 180.0)
        
        df_clean = df[valid_lat & valid_lon].copy()
        
        invalid_count = initial_len - len(df_clean)
        if invalid_count > 0:
            logger.warning("Dropped %d rows in %s with invalid coordinates (out of geographic bounds).", 
                           invalid_count, dataset_name)
            
        return df_clean

    def ingest_all(self, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes ingestion pipeline for all configured raw data files.
        """
        paths = config["data"]
        
        # Ingest and validate
        df_dispatch = self.load_csv(paths["dispatch_file"], "dispatch")
        self.validate_schema(df_dispatch, "dispatch")
        df_dispatch = self.clean_duplicates(df_dispatch, "dispatch")
        
        df_gps = self.load_csv(paths["gps_file"], "gps")
        self.validate_schema(df_gps, "gps")
        df_gps = self.clean_duplicates(df_gps, "gps")
        df_gps = self.validate_coordinates(df_gps, "gps")
        
        df_traffic = self.load_csv(paths["traffic_file"], "traffic")
        self.validate_schema(df_traffic, "traffic")
        df_traffic = self.clean_duplicates(df_traffic, "traffic")
        
        df_weather = self.load_csv(paths["weather_file"], "weather")
        self.validate_schema(df_weather, "weather")
        df_weather = self.clean_duplicates(df_weather, "weather")
        
        return df_dispatch, df_gps, df_traffic, df_weather
