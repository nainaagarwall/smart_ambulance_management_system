# Intelligent Emergency Response Optimization System (EROS)

EROS is a Python-based predictive analytics platform designed to optimize urban ambulance dispatching and routing decisions. It ingests and validates heterogeneous datasets—including real-time GPS streams, historical emergency dispatch logs, hourly traffic congestion records, and weather conditions—engineers predictive features, and trains machine learning models to anticipate response delays, classify optimal standby zones, and forecast arrival time categories.

---

## System Architecture

The platform is designed with a modular, layered architecture:

```text
intelligent_emergency_response/
│
├── config/
│   └── config.yaml               # Configurable thresholds, parameters, and hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader and logging setup
│   ├── data_generator.py         # Synthetic data generator modeling NYC-ish environments
│   ├── ingestion.py              # Schema validation, duplicates clean, bounds check
│   ├── preprocessing.py          # GPS interpolation, timestamp repair, spatial-temporal merges
│   ├── feature_engineering.py    # Vectorized computation of 10 predictive features
│   ├── models.py                 # Multi-target model training, evaluation, and auto-selection
│   ├── visualization.py          # Saves heatmaps, trajectories, hotspots, and trends
│   ├── reporting.py              # Exports JSON dispatch alerts and CSV admin summaries
│   └── pipeline.py               # End-to-end pipeline orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py     # Unit tests for preprocessing & cleaning
│   └── test_pipeline.py          # Integration tests for end-to-end pipeline
│
├── run.py                        # CLI entrypoint to execute the pipeline
├── README.md                     # Setup and execution instructions
└── pyproject.toml                # Project metadata and dependencies managed by uv
```

---

## Core Features & Requirements Met

1.  **Data Processing & Cleaning**:
    *   **Timestamp Repair**: Identifies and repairs inverted timestamps (arrival before dispatch) or missing arrival durations using median emergency-type metrics.
    *   **GPS Interpolation**: Fills missing GPS latitude/longitude dropouts using linear interpolation.
    *   **Outlier Removal**: Filters travel speeds > 150 km/h and response durations > 120 minutes.
    *   **Spatial-Temporal Merge**: Maps incident locations to nearest road segments. Merges weather and traffic data hourly and tracks average actual travel speeds from GPS tracking streams.
    *   **Encoding & Normalization**: Standardizes continuous fields using `StandardScaler` and One-Hot encodes categorical columns.

2.  **Advanced Feature Engineering**:
    *   *Average Congestion during Dispatch*: Congestion index mapping at the time of dispatch.
    *   *Estimated Travel Delay*: Difference between actual response duration and normal free-flow travel time.
    *   *Distance-to-Incident Ratio*: Ratio of actual route length (summed from GPS steps) to straight-line Haversine distance (route windingness).
    *   *Peak-Hour Emergency Frequency*: Historical incident count ratios by hour of the day.
    *   *Historical Ambulance Response Efficiency*: Historical average response duration by vehicle.
    *   *Weather-Adjusted Travel Speed*: Speeds scaled down based on storm level and rain intensity.
    *   *Route Risk Score*: Composite score derived from accident reports, road closures, storm levels, and congestion levels.
    *   *Emergency Severity Index*: Map of emergencies to a numeric index (1 to 5).
    *   *Rolling Traffic Congestion Trends*: 3-hour rolling average congestion level per segment.
    *   *Nearby Hospital Load Indicators*: Count of active dispatches heading to the destination hospital in a rolling window.

3.  **Machine Learning & Evaluation**:
    *   Trains both **Random Forest** and **Gradient Boosting** models.
    *   Predicts three distinct target variables:
        1.  `is_delayed` (Binary: response > 15 mins)
        2.  `optimal_zone` (Multiclass: geographic deployment grid zones)
        3.  `arrival_category` (Multiclass: Fast `<10m`, Standard `10-20m`, Delayed `>20m`)
    *   Calculates **ROC-AUC, Precision, Recall, F1-Score, and Confusion Matrices**.
    *   Automatically selects the best model for each target and saves them as `.joblib` files.

4.  **Reporting**:
    *   **JSON Alerts**: High-risk dispatcher warnings (probability > 70%) containing recommended actions.
    *   **CSV Summaries**: Tabulates emergency volumes, response averages, and recommended ambulance standby count by zone.

5.  **Visualizations**:
    *   `traffic_heatmap.png`: Segment locations colored by mean congestion.
    *   `ambulance_trajectories.png`: GPS path trajectories of fleet vehicles.
    *   `emergency_hotspots.png`: KDE plot of historical incident coordinate density.
    *   `prediction_probabilities.png`: Predicted delay probabilities distribution.
    *   `hourly_response_trends.png`: Hourly response durations line plot.

---

## Setup Instructions

This project uses the fast Python packaging tool **`uv`**.

1.  **Clone or create the project directory**:
    Ensure you are inside the directory:
    ```powershell
    cd C:\Users\Admin\.gemini\antigravity-ide\scratch\intelligent_emergency_response
    ```

2.  **Install dependencies**:
    `uv` will automatically download Python 3.12 and create a virtual environment:
    ```powershell
    uv sync
    ```

---

## Execution Instructions

Run the complete pipeline end-to-end, including generating a fresh synthetic dataset:
```powershell
uv run run.py --config config/config.yaml
```

### CLI Options
*   `--config <path>`: Specify a custom configuration file.
*   `--no-generate`: Skip dataset generation and run on existing files in `data/raw/`.
*   `--debug`: Print verbose debugging logs.

For example, to run with debug outputs:
```powershell
uv run run.py --debug
```

---

## Testing

EROS includes unit and integration tests covering cleaning operations and end-to-end prediction flows:
```powershell
uv run pytest
```
All tests should pass.
