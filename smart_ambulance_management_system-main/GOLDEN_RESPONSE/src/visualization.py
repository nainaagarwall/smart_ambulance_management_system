"""
Visualization Layer for the Intelligent Emergency Response Optimization System.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResponseVisualizer:
    """
    Generates high-quality spatial and statistical plots for response analytics.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.plots_dir = config["data"].get("plots_dir", "outputs/plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Apply modern style parameters
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("talk")
        
        self.primary_color = "#1e3d59"  # Deep blue
        self.accent_color = "#ff6e40"   # Vibrant orange
        self.bg_color = "#f5f0e1"       # Soft cream
        self.palette = sns.color_palette("muted")

    def plot_traffic_heatmap(self, df_traffic: pd.DataFrame, df_gps: pd.DataFrame) -> str:
        """
        Creates a traffic congestion heatmap.
        Maps road segments spatially based on their GPS coordinates and colors them by congestion level.
        Uses geopandas to create a spatial DataFrame of road segments.
        """
        logger.info("Generating traffic congestion heatmap...")
        
        # Find mean coordinates of each segment from GPS
        segment_coords = df_gps.dropna(subset=["latitude", "longitude"]).groupby("route_segment_id")[["latitude", "longitude"]].mean().reset_index()
        segment_coords = segment_coords.rename(columns={"route_segment_id": "road_segment_id"})
        
        # Merge with traffic congestion data (mean congestion index per segment)
        congestion_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Severe": 4.0}
        df_traffic_copy = df_traffic.copy()
        df_traffic_copy["congestion_index"] = df_traffic_copy["congestion_level"].map(congestion_map).fillna(1.0)
        
        segment_congestion = df_traffic_copy.groupby("road_segment_id")["congestion_index"].mean().reset_index()
        df_merged = pd.merge(segment_coords, segment_congestion, on="road_segment_id", how="inner")
        
        # Use Geopandas to build a spatial dataset
        geometry = [Point(xy) for xy in zip(df_merged["longitude"], df_merged["latitude"])]
        gdf = gpd.GeoDataFrame(df_merged, geometry=geometry, crs="EPSG:4326")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot with geopandas
        scatter = gdf.plot(
            ax=ax,
            column="congestion_index",
            cmap="YlOrRd",
            legend=True,
            legend_kwds={"label": "Mean Congestion Index (1=Low, 4=Severe)", "orientation": "horizontal"},
            markersize=300,
            alpha=0.8,
            edgecolor="grey",
            linewidth=1.2
        )
        
        # Add labels to segments
        for idx, row in gdf.iterrows():
            ax.annotate(
                text=row["road_segment_id"],
                xy=(row["geometry"].x, row["geometry"].y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                weight="bold"
            )
            
        ax.set_title("Urban Road Segment Congestion Heatmap", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        
        save_path = os.path.join(self.plots_dir, "traffic_heatmap.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info("Saved traffic congestion heatmap to %s", save_path)
        return save_path

    def plot_ambulance_trajectory(self, df_gps: pd.DataFrame, num_ambulances: int = 3) -> str:
        """
        Plots movement trajectories of active ambulances based on their GPS tracking coordinates.
        Creates path lines using geopandas and shapely.
        """
        logger.info("Generating ambulance trajectory plot...")
        
        # Select ambulances with the most GPS records
        top_ambulances = df_gps["ambulance_id"].value_counts().head(num_ambulances).index.tolist()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Color list
        colors = ["#1e3d59", "#ff6e40", "#17b978"]
        
        for idx, amb_id in enumerate(top_ambulances):
            amb_data = df_gps[df_gps["ambulance_id"] == amb_id].sort_values(by="timestamp")
            
            # Drop NaN coordinates
            amb_data = amb_data.dropna(subset=["latitude", "longitude"])
            if len(amb_data) < 2:
                continue
                
            # Create a line string geometry for the trajectory
            points = [Point(lon, lat) for lon, lat in zip(amb_data["longitude"], amb_data["latitude"])]
            line = LineString(points)
            
            # Plot line
            x, y = line.xy
            ax.plot(x, y, label=f"Ambulance {amb_id}", color=colors[idx % len(colors)], linewidth=2.5, marker='o', markersize=3, alpha=0.8)
            
            # Start and end indicators
            ax.scatter(x[0], y[0], color="green", marker="^", s=150, zorder=5, label="Start Point" if idx == 0 else "")
            ax.scatter(x[-1], y[-1], color="red", marker="X", s=150, zorder=5, label="End Point" if idx == 0 else "")
            
        ax.set_title(f"Ambulance Movement Trajectories (Top {num_ambulances} Fleet Vehicles)", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.legend(loc="best", frameon=True)
        
        save_path = os.path.join(self.plots_dir, "ambulance_trajectories.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info("Saved ambulance trajectories to %s", save_path)
        return save_path

    def plot_emergency_hotspots(self, df_dispatch: pd.DataFrame) -> str:
        """
        Generates emergency hotspot analysis using a 2D KDE (Kernel Density Estimation) plot.
        """
        logger.info("Generating emergency hotspot density map...")
        
        df = df_dispatch.copy()
        # Extract lat/lon
        df[["latitude", "longitude"]] = df["incident_location"].str.split(",", expand=True).astype(float)
        df = df.dropna(subset=["latitude", "longitude"])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Kernel density estimation plot
        sns.kdeplot(
            data=df,
            x="longitude",
            y="latitude",
            cmap="Reds",
            fill=True,
            thresh=0.05,
            levels=15,
            alpha=0.6,
            ax=ax,
            cbar=True,
            cbar_kws={"label": "Density of Emergency Calls"}
        )
        
        # Overlay actual incident points as small dots
        ax.scatter(df["longitude"], df["latitude"], color="black", s=5, alpha=0.3, label="Incident Locations")
        
        ax.set_title("Emergency Response Incident Hotspot Analysis", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.legend(loc="upper right", frameon=True)
        
        save_path = os.path.join(self.plots_dir, "emergency_hotspots.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info("Saved emergency hotspots to %s", save_path)
        return save_path

    def plot_prediction_probability(self, y_prob: np.ndarray) -> str:
        """
        Plots prediction probability distribution for delayed emergency responses.
        """
        logger.info("Generating prediction probability distribution chart...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram and KDE
        sns.histplot(y_prob, kde=True, color="#ff6e40", ax=ax, bins=25, stat="probability")
        
        # Add risk threshold marker (0.7)
        ax.axvline(x=0.7, color="#1e3d59", linestyle="--", linewidth=2.5, label="High-Risk Threshold (0.70)")
        
        ax.set_title("Distribution of Predicted Delayed-Response Probabilities", fontsize=16, weight="bold", pad=15)
        ax.set_xlabel("Predicted Probability of Delay", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(loc="best", frameon=True)
        
        save_path = os.path.join(self.plots_dir, "prediction_probabilities.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info("Saved prediction probabilities chart to %s", save_path)
        return save_path

    def plot_hourly_trends(self, df_dispatch: pd.DataFrame) -> str:
        """
        Plots hourly response-time trend graphs.
        """
        logger.info("Generating hourly response-time trend line plot...")
        
        df = df_dispatch.copy()
        df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
        df["hour"] = df["dispatch_time"].dt.hour
        
        hourly_stats = df.groupby("hour")["response_duration_minutes"].agg(["mean", "std", "count"]).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Mean line
        ax.plot(hourly_stats["hour"], hourly_stats["mean"], color="#1e3d59", marker="o", linewidth=2.5, label="Mean Response Time")
        
        # Standard deviation confidence band
        lower_bound = (hourly_stats["mean"] - 0.5 * hourly_stats["std"]).clip(lower=0)
        upper_bound = hourly_stats["mean"] + 0.5 * hourly_stats["std"]
        ax.fill_between(
            hourly_stats["hour"], 
            lower_bound, 
            upper_bound, 
            color="#1e3d59", 
            alpha=0.15, 
            label="Response Variability (±0.5 SD)"
        )
        
        ax.set_title("Hourly Emergency Response Time Trends", fontsize=16, weight="bold", pad=15)
        ax.set_xlabel("Hour of Day (24-Hour Format)", fontsize=12)
        ax.set_ylabel("Response Duration (Minutes)", fontsize=12)
        ax.set_xticks(range(0, 24))
        ax.set_xlim(-0.5, 23.5)
        ax.legend(loc="best", frameon=True)
        
        save_path = os.path.join(self.plots_dir, "hourly_response_trends.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info("Saved hourly response trends to %s", save_path)
        return save_path
        
    def generate_all_plots(self, df_dispatch: pd.DataFrame, df_gps: pd.DataFrame, df_traffic: pd.DataFrame, y_prob: np.ndarray) -> None:
        """
        Generates and saves all five required visualizations.
        """
        self.plot_traffic_heatmap(df_traffic, df_gps)
        self.plot_ambulance_trajectory(df_gps)
        self.plot_emergency_hotspots(df_dispatch)
        self.plot_prediction_probability(y_prob)
        self.plot_hourly_trends(df_dispatch)
