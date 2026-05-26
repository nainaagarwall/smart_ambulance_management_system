"""
Synthetic dataset generator for testing the Intelligent Emergency Response Optimization System.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(output_dir: str = "data/raw", seed: int = 42) -> None:
    """
    Generates synthetic datasets modeling an urban emergency response environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    start_time = datetime(2026, 5, 18, 0, 0, 0)
    end_time = datetime(2026, 5, 24, 23, 59, 59)
    num_incidents = 1000
    num_ambulances = 20
    num_segments = 50
    
    # Bounds for Latitude and Longitude (NYC-ish)
    lat_min, lat_max = 40.7000, 40.8500
    lon_min, lon_max = -74.0200, -73.9000
    
    # 1. Road segments definitions
    segments = []
    for i in range(1, num_segments + 1):
        seg_id = f"SEG_{i:03d}"
        # Central coordinate of the segment
        seg_lat = random.uniform(lat_min, lat_max)
        seg_lon = random.uniform(lon_min, lon_max)
        segments.append({"road_segment_id": seg_id, "lat": seg_lat, "lon": seg_lon})
    df_segments = pd.DataFrame(segments)
    
    # 2. Hospitals definitions
    hospitals = [
        {"name": "City Hospital", "lat": 40.7128, "lon": -74.0060},
        {"name": "St. Jude Medical Center", "lat": 40.7589, "lon": -73.9851},
        {"name": "General Infirmary", "lat": 40.7829, "lon": -73.9654},
        {"name": "Mercy Hospital", "lat": 40.8068, "lon": -73.9568}
    ]
    
    emergency_types = ["Cardiac Arrest", "Stroke", "Trauma", "Respiratory Distress", "Minor Injury"]
    
    # Generate Weather Data (hourly)
    weather_records = []
    curr_time = start_time
    while curr_time <= end_time:
        rainfall = 0.0
        if random.random() < 0.2:  # 20% chance of rain
            rainfall = round(random.uniform(1.0, 25.0), 1)
        
        storm_alert_level = 0
        if rainfall > 15.0:
            storm_alert_level = 2
        elif rainfall > 5.0:
            storm_alert_level = 1
            
        visibility = round(random.uniform(8.0, 10.0), 1)
        if rainfall > 0:
            visibility = round(max(1.0, visibility - rainfall * 0.3), 1)
            
        temp = round(random.uniform(15.0, 30.0), 1)
        
        weather_records.append({
            "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
            "rainfall": rainfall,
            "visibility": visibility,
            "temperature": temp,
            "storm_alert_level": storm_alert_level
        })
        curr_time += timedelta(hours=1)
        
    df_weather = pd.DataFrame(weather_records)
    df_weather.to_csv(os.path.join(output_dir, "weather_data.csv"), index=False)
    
    # Generate Traffic Monitoring Data (hourly for each segment)
    traffic_records = []
    curr_time = start_time
    while curr_time <= end_time:
        hour = curr_time.hour
        # Peak traffic hours (7-9 AM, 4-7 PM)
        is_peak = (7 <= hour <= 9) or (16 <= hour <= 19)
        
        for seg in segments:
            seg_id = seg["road_segment_id"]
            
            # Base congestion logic
            prob = random.random()
            if is_peak:
                congestion_level = random.choices(["Low", "Medium", "High", "Severe"], weights=[10, 30, 40, 20])[0]
            else:
                congestion_level = random.choices(["Low", "Medium", "High", "Severe"], weights=[60, 30, 8, 2])[0]
                
            base_speed = 50.0  # km/h
            speed_map = {"Low": 1.0, "Medium": 0.7, "High": 0.4, "Severe": 0.2}
            avg_speed = round(base_speed * speed_map[congestion_level] * random.uniform(0.9, 1.1), 1)
            
            accident_reports = 1 if (random.random() < 0.02) else 0
            road_closure_status = "Closed" if (random.random() < 0.01) else "Open"
            
            traffic_records.append({
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "road_segment_id": seg_id,
                "congestion_level": congestion_level,
                "average_vehicle_speed": avg_speed,
                "accident_reports": accident_reports,
                "road_closure_status": road_closure_status
            })
        curr_time += timedelta(hours=1)
        
    df_traffic = pd.DataFrame(traffic_records)
    df_traffic.to_csv(os.path.join(output_dir, "traffic_monitoring.csv"), index=False)
    
    # Generate Emergency Dispatch Records and corresponding GPS traces
    dispatch_records = []
    gps_records = []
    
    for inc_idx in range(1, num_incidents + 1):
        inc_id = f"INC_{inc_idx:04d}"
        amb_id = f"AMB_{random.randint(1, num_ambulances):03d}"
        
        # Dispatch time within range
        seconds_offset = random.randint(0, int((end_time - start_time).total_seconds() - 7200))
        d_time = start_time + timedelta(seconds=seconds_offset)
        
        # Select target hospital and incident coordinates
        hosp = random.choice(hospitals)
        hosp_dest = hosp["name"]
        
        inc_lat = random.uniform(lat_min, lat_max)
        inc_lon = random.uniform(lon_min, lon_max)
        inc_loc = f"{inc_lat:.5f},{inc_lon:.5f}"
        
        etype = random.choice(emergency_types)
        
        # Calculate distance to incident from hospital (acting as dispatch base)
        # Haversine approximation
        dist = 111 * np.sqrt((inc_lat - hosp["lat"])**2 + (inc_lon - hosp["lon"])**2)
        
        # Compute baseline travel duration (minutes)
        # Factor in distance, weather, traffic
        weather_at_time = df_weather.iloc[int((d_time - start_time).total_seconds() // 3600)]
        storm = weather_at_time["storm_alert_level"]
        
        # Approximate average congestion speed modifier
        # Pick a random segment to match
        matched_seg = df_segments.iloc[random.randint(0, num_segments - 1)]
        traffic_hour = df_traffic[
            (df_traffic["timestamp"] == d_time.replace(minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")) & 
            (df_traffic["road_segment_id"] == matched_seg["road_segment_id"])
        ]
        
        if not traffic_hour.empty:
            speed_mod = traffic_hour.iloc[0]["average_vehicle_speed"] / 50.0
        else:
            speed_mod = 1.0
            
        weather_mod = 1.0 - (0.15 * storm)
        effective_speed = max(10.0, 45.0 * speed_mod * weather_mod) # in km/h
        
        # response duration in minutes
        duration = (dist / effective_speed) * 60.0 + random.uniform(2.0, 8.0)
        
        # Inject corrupted records / outliers to test preprocessing
        is_corrupted = False
        if random.random() < 0.03:  # 3% corrupted timestamps
            is_corrupted = True
            corrupt_type = random.randint(1, 3)
            if corrupt_type == 1:
                # arrival before dispatch
                a_time = d_time - timedelta(minutes=5)
                duration = -5.0
            elif corrupt_type == 2:
                # missing arrival time
                a_time = None
                duration = np.nan
            else:
                # arrival time way in the future (outlier)
                duration = 180.0
                a_time = d_time + timedelta(minutes=duration)
        else:
            duration = max(2.0, round(duration, 1))
            a_time = d_time + timedelta(minutes=duration)
            
        dispatch_records.append({
            "incident_id": inc_id,
            "ambulance_id": amb_id,
            "dispatch_time": d_time.strftime("%Y-%m-%d %H:%M:%S"),
            "arrival_time": a_time.strftime("%Y-%m-%d %H:%M:%S") if a_time else np.nan,
            "incident_location": inc_loc,
            "hospital_destination": hosp_dest,
            "emergency_type": etype,
            "response_duration_minutes": duration
        })
        
        # If not completely corrupted, generate GPS trace during this run
        if not is_corrupted and a_time:
            # Generate GPS samples every minute
            steps = int(np.ceil(duration))
            hosp_lat, hosp_lon = hosp["lat"], hosp["lon"]
            
            for step in range(steps + 1):
                t = d_time + timedelta(minutes=step)
                if t > a_time:
                    t = a_time
                    
                fraction = step / steps if steps > 0 else 1.0
                curr_lat = hosp_lat + (inc_lat - hosp_lat) * fraction
                curr_lon = hosp_lon + (inc_lon - hosp_lon) * fraction
                
                # GPS Noise
                curr_lat += random.normalvariate(0, 0.0001)
                curr_lon += random.normalvariate(0, 0.0001)
                
                # Check for GPS dropout / missing coordinates
                if random.random() < 0.05:  # 5% chance of missing GPS
                    curr_lat = np.nan
                    curr_lon = np.nan
                
                speed = effective_speed * random.uniform(0.8, 1.2)
                # Inject speed outliers
                if random.random() < 0.005:
                    speed = 220.0  # unrealistic speed
                    
                # Match current segment based on closest distance
                distances = (df_segments["lat"] - curr_lat)**2 + (df_segments["lon"] - curr_lon)**2
                closest_seg_id = df_segments.iloc[distances.argmin()]["road_segment_id"] if not np.isnan(curr_lat) else "SEG_001"
                
                gps_records.append({
                    "ambulance_id": amb_id,
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "latitude": curr_lat,
                    "longitude": curr_lon,
                    "current_speed": round(speed, 1),
                    "route_segment_id": closest_seg_id
                })
                
    # Save dispatch
    df_dispatch = pd.DataFrame(dispatch_records)
    # Inject duplicate records
    duplicates = df_dispatch.sample(n=20, random_state=42)
    df_dispatch = pd.concat([df_dispatch, duplicates], ignore_index=True)
    df_dispatch.to_csv(os.path.join(output_dir, "dispatch_records.csv"), index=False)
    
    # Save GPS
    df_gps = pd.DataFrame(gps_records)
    df_gps.to_csv(os.path.join(output_dir, "gps_tracking.csv"), index=False)
    
    print(f"Generated synthetic datasets in '{output_dir}':")
    print(f"  - Dispatch records: {len(df_dispatch)} rows")
    print(f"  - GPS Tracking data: {len(df_gps)} rows")
    print(f"  - Traffic data: {len(df_traffic)} rows")
    print(f"  - Weather data: {len(df_weather)} rows")
    
if __name__ == "__main__":
    generate_synthetic_data()
