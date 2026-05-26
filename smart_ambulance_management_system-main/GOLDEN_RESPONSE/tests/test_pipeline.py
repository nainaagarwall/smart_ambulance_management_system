"""
Integration tests for the end-to-end Emergency Response prediction pipeline.
"""

import os
import shutil
import tempfile
import yaml
import pytest
from src.pipeline import EmergencyResponsePipeline
from src.data_generator import generate_synthetic_data

@pytest.fixture
def temp_project_dir():
    """
    Creates a temporary workspace directory for pipeline tests.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_pipeline_integration(temp_project_dir):
    """
    Runs the entire pipeline end-to-end on a minimal dataset to verify
    data flow and model selection logic.
    """
    # 1. Create a custom test config.yaml in the temp directory
    config_dict = {
        "system": {
            "random_seed": 42,
            "log_level": "WARNING"
        },
        "data": {
            "raw_dir": os.path.join(temp_project_dir, "data/raw"),
            "output_dir": os.path.join(temp_project_dir, "outputs"),
            "reports_dir": os.path.join(temp_project_dir, "outputs/reports"),
            "plots_dir": os.path.join(temp_project_dir, "outputs/plots"),
            "models_dir": os.path.join(temp_project_dir, "outputs/models"),
            "dispatch_file": os.path.join(temp_project_dir, "data/raw/dispatch_records.csv"),
            "gps_file": os.path.join(temp_project_dir, "data/raw/gps_tracking.csv"),
            "traffic_file": os.path.join(temp_project_dir, "data/raw/traffic_monitoring.csv"),
            "weather_file": os.path.join(temp_project_dir, "data/raw/weather_data.csv")
        },
        "preprocessing": {
            "max_allowable_speed": 150.0,
            "min_allowable_speed": 2.0,
            "max_response_duration": 120.0,
            "min_response_duration": 1.0,
            "gps_interpolation_limit": 5,
            "corrupted_time_fallback_minutes": 15.0
        },
        "features": {
            "rolling_congestion_window": "3h",
            "heavy_rain_threshold": 10.0,
            "low_visibility_threshold": 5.0,
            "delayed_response_threshold_minutes": 15.0,
            "spatial_grid_size_km": 1.0
        },
        "models": {
            "random_forest": {
                "n_estimators": 5,  # Very small for speed
                "max_depth": 3,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 5,  # Very small for speed
                "max_depth": 2,
                "random_state": 42
            },
            "test_size": 0.2
        }
    }
    
    config_file_path = os.path.join(temp_project_dir, "test_config.yaml")
    with open(config_file_path, "w") as f:
        yaml.dump(config_dict, f)
        
    # 2. Pre-generate a small dataset to verify pipeline runs on it
    # We copy generator but run it with fewer records if possible, or use default generator
    # For speed, let's run the generator on the custom raw dir
    # Note: we call our synthetic generator
    generate_synthetic_data(config_dict["data"]["raw_dir"], seed=42)
    
    # 3. Initialize pipeline with test config and execute
    pipeline = EmergencyResponsePipeline(config_path=config_file_path)
    eval_report, summary_path = pipeline.run_pipeline(generate_data=False)
    
    # 4. Verify outputs are generated
    assert os.path.exists(summary_path)
    assert os.path.exists(os.path.join(config_dict["data"]["reports_dir"], "delayed_response_alerts.json"))
    assert os.path.exists(os.path.join(config_dict["data"]["reports_dir"], "city_admin_analytics_summary.csv"))
    
    # Check that model objects are saved
    assert os.path.exists(os.path.join(config_dict["data"]["models_dir"], "is_delayed_best_model.joblib"))
    assert os.path.exists(os.path.join(config_dict["data"]["models_dir"], "optimal_zone_best_model.joblib"))
    assert os.path.exists(os.path.join(config_dict["data"]["models_dir"], "arrival_category_best_model.joblib"))
    
    # Check that plots are created
    plots_dir = config_dict["data"]["plots_dir"]
    assert os.path.exists(os.path.join(plots_dir, "traffic_heatmap.png"))
    assert os.path.exists(os.path.join(plots_dir, "ambulance_trajectories.png"))
    assert os.path.exists(os.path.join(plots_dir, "emergency_hotspots.png"))
    assert os.path.exists(os.path.join(plots_dir, "prediction_probabilities.png"))
    assert os.path.exists(os.path.join(plots_dir, "hourly_response_trends.png"))
    
    # Verify that eval_report contains accuracy scores for all targets
    for target in ["is_delayed", "optimal_zone", "arrival_category"]:
        assert target in eval_report
        assert "random_forest" in eval_report[target]
        assert "gradient_boosting" in eval_report[target]
        assert "accuracy" in eval_report[target]["random_forest"]
