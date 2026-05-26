"""
Unit tests for data preprocessing and cleaning components.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import DataPreprocessor

@pytest.fixture
def sample_config():
    return {
        "preprocessing": {
            "max_allowable_speed": 150.0,
            "min_allowable_speed": 2.0,
            "max_response_duration": 120.0,
            "min_response_duration": 1.0,
            "gps_interpolation_limit": 5,
            "corrupted_time_fallback_minutes": 15.0
        }
    }

def test_repair_timestamps(sample_config):
    preprocessor = DataPreprocessor(sample_config)
    
    # Create sample dispatch data with corrupted timestamps
    df_dispatch = pd.DataFrame({
        "incident_id": ["INC_001", "INC_002", "INC_003"],
        "emergency_type": ["Cardiac Arrest", "Minor Injury", "Cardiac Arrest"],
        "dispatch_time": ["2026-05-18 10:00:00", "2026-05-18 11:00:00", "2026-05-18 12:00:00"],
        "arrival_time": [
            "2026-05-18 09:55:00",  # Inverted (arrival before dispatch)
            np.nan,                 # Missing arrival
            "2026-05-18 12:10:00"   # Valid
        ],
        "response_duration_minutes": [-5.0, np.nan, 10.0]
    })
    
    df_repaired = preprocessor.repair_timestamps(df_dispatch)
    
    # Assertions
    assert len(df_repaired) == 3
    # Check that arrival times are repaired and are after dispatch times
    assert (df_repaired["arrival_time"] >= df_repaired["dispatch_time"]).all()
    # Check that durations are positive
    assert (df_repaired["response_duration_minutes"] > 0).all()
    # INC_003 was valid, check its duration was preserved
    assert df_repaired.loc[2, "response_duration_minutes"] == 10.0

def test_interpolate_gps_coordinates(sample_config):
    preprocessor = DataPreprocessor(sample_config)
    
    # Create sample GPS data with missing coordinates
    df_gps = pd.DataFrame({
        "ambulance_id": ["AMB_001", "AMB_001", "AMB_001", "AMB_001", "AMB_001"],
        "timestamp": [
            "2026-05-18 10:00:00", "2026-05-18 10:01:00", "2026-05-18 10:02:00", 
            "2026-05-18 10:03:00", "2026-05-18 10:04:00"
        ],
        "latitude": [40.7, np.nan, 40.8, np.nan, 41.0],
        "longitude": [-74.0, np.nan, -74.1, np.nan, -74.3],
        "current_speed": [30.0, 35.0, 40.0, 32.0, 45.0],
        "route_segment_id": ["SEG_001", "SEG_001", "SEG_002", "SEG_002", "SEG_003"]
    })
    
    df_interpolated = preprocessor.interpolate_gps_coordinates(df_gps)
    
    # Assertions
    assert len(df_interpolated) == 5
    # The NaNs (index 1 and 3) should be interpolated
    assert not df_interpolated["latitude"].isna().any()
    assert not df_interpolated["longitude"].isna().any()
    # Check linear interpolation math: (40.7 + 40.8)/2 = 40.75
    assert df_interpolated.loc[1, "latitude"] == pytest.approx(40.75)
    assert df_interpolated.loc[3, "latitude"] == pytest.approx(40.9)

def test_remove_outliers(sample_config):
    preprocessor = DataPreprocessor(sample_config)
    
    # Dispatch outliers
    df_dispatch = pd.DataFrame({
        "incident_id": ["INC_001", "INC_002", "INC_003"],
        "response_duration_minutes": [5.0, 150.0, 10.0] # 150 is > max_response_duration (120)
    })
    
    # GPS outliers
    df_gps = pd.DataFrame({
        "ambulance_id": ["AMB_001", "AMB_001", "AMB_001"],
        "current_speed": [40.0, 200.0, 10.0] # 200 is > max_allowable_speed (150)
    })
    
    df_disp_clean, df_gps_clean = preprocessor.remove_outliers(df_dispatch, df_gps)
    
    # Assertions
    # INC_002 should be removed (duration 150)
    assert len(df_disp_clean) == 2
    assert "INC_002" not in df_disp_clean["incident_id"].values
    
    # Speed 200 should be removed
    assert len(df_gps_clean) == 2
    assert 200.0 not in df_gps_clean["current_speed"].values
