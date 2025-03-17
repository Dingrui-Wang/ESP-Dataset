import json
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point, MultiLineString
from shapely.ops import linemerge
import pandas as pd
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.patches as patches


class MapVisualizer:
    """A class to visualize map data from ESP dataset JSON files."""
    
    def __init__(self, map_path=None):
        """
        Initialize the map visualizer.
        
        Args:
            map_path (str, optional): Path to a specific map JSON file. If None, no map is loaded.
        """
        self.map_data = None
        self.lane_gdf = None
        if map_path:
            self.load_map(map_path)
    
    def load_map(self, map_path):
        """
        Load map data from a JSON file.
        
        Args:
            map_path (str): Path to the map JSON file.
        """
        with open(map_path, 'r') as f:
            self.map_data = json.load(f)
        
        self.map_id = Path(map_path).stem
        print(f"Loaded map: {self.map_id}")
        
        # Check map structure
        if "lanes" in self.map_data:
            print(f"Map contains {len(self.map_data['lanes'])} lanes")
            self.lane_gdf = self.extract_lane_segments()
        else:
            print("Warning: No lanes found in map data")
            self.lane_gdf = gpd.GeoDataFrame(columns=['geometry', 'type', 'id', 'length', 'num_points'])
            
        return self.map_data
    
    def extract_lane_segments(self):
        """
        Extract lane segments from the map data.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing lane segments.
        """
        if not self.map_data:
            raise ValueError("No map data loaded. Call load_map first.")
        
        lanes = []           # Geometries
        lane_types = []      # Type of lane feature
        lane_ids = []        # ID for each lane feature
        lane_lengths = []    # Length of each lane feature
        point_counts = []    # Number of points in each feature
        
        # Process road information from road_infos
        if "road_infos" in self.map_data:
            for road_idx, road_info in enumerate(self.map_data.get("road_infos", [])):
                if "adas_infos" in road_info:
                    road_points = []
                    for adas_info in road_info["adas_infos"]:
                        if "pt" in adas_info:
                            road_points.append((adas_info["pt"]["x"], adas_info["pt"]["y"]))
                    
                    if len(road_points) >= 2:
                        road_geom = LineString(road_points)
                        lanes.append(road_geom)
                        lane_types.append("road_center")
                        lane_ids.append(f"road_{road_idx}")
                        lane_lengths.append(road_geom.length)
                        point_counts.append(len(road_points))
        
        # Process lane cone data if available
        for lane_idx, lane in enumerate(self.map_data.get("lanes", [])):
            lane_id = lane.get("global_lane_id", lane_idx)
            
            # Check for cone_infos
            if "cone_infos" in lane and lane["cone_infos"]:
                for side_idx, side in enumerate(["left", "right"]):
                    cone_points = []
                    for cone in lane["cone_infos"]:
                        if side in cone and "x" in cone[side] and "y" in cone[side]:
                            cone_points.append((cone[side]["x"], cone[side]["y"]))
                    
                    if len(cone_points) >= 2:
                        cone_geom = LineString(cone_points)
                        lanes.append(cone_geom)
                        lane_types.append(f"{side}_cone")
                        lane_ids.append(f"{side}_cone_{lane_id}")
                        lane_lengths.append(cone_geom.length)
                        point_counts.append(len(cone_points))
        
        # If we have lanes in the GeoDataFrame, return it
        if lanes:
            lane_gdf = gpd.GeoDataFrame({
                'geometry': lanes,
                'type': lane_types,
                'id': lane_ids,
                'length': lane_lengths,
                'num_points': point_counts
            })
            
            print(f"Extracted {len(lane_gdf)} lane segments")
            print(f"Lane types: {lane_gdf['type'].unique()}")
            return lane_gdf
        
        # If no lanes, create dummy lane data based on first road segment point
        if "road_infos" in self.map_data and self.map_data["road_infos"]:
            first_road = self.map_data["road_infos"][0]
            if "adas_infos" in first_road and first_road["adas_infos"]:
                first_point = first_road["adas_infos"][0]
                if "pt" in first_point:
                    x, y = first_point["pt"]["x"], first_point["pt"]["y"]
                    center_line = LineString([(x-100, y), (x+100, y)])
                    lanes.append(center_line)
                    lane_types.append("reference_line")
                    lane_ids.append("reference")
                    lane_lengths.append(200)  # 200m line
                    point_counts.append(2)
        
        # If still no data, return empty DataFrame
        if not lanes:
            return gpd.GeoDataFrame(columns=['geometry', 'type', 'id', 'length', 'num_points'])
        
        lane_gdf = gpd.GeoDataFrame({
            'geometry': lanes,
            'type': lane_types,
            'id': lane_ids,
            'length': lane_lengths,
            'num_points': point_counts
        })
        
        print(f"Extracted {len(lane_gdf)} lane segments")
        return lane_gdf
    
    def get_map_bounds(self, buffer=50):
        """
        Get the bounds of the map with a buffer.
        
        Args:
            buffer (float): Buffer distance to add around the bounds.
            
        Returns:
            tuple: (xmin, ymin, xmax, ymax) bounds with buffer.
        """
        # Handle empty lane data
        if self.lane_gdf is None or self.lane_gdf.empty:
            # Try to get bounds from a road point
            if "road_infos" in self.map_data and self.map_data["road_infos"]:
                for road in self.map_data["road_infos"]:
                    if "adas_infos" in road and road["adas_infos"]:
                        for adas_info in road["adas_infos"]:
                            if "pt" in adas_info:
                                x, y = adas_info["pt"]["x"], adas_info["pt"]["y"]
                                return (x - 500, y - 500, x + 500, y + 500)
            
            # Default bounds if no other data available
            return (-1000, -1000, 1000, 1000)
        
        # If we have lane data, use it for bounds
        total_bounds = self.lane_gdf.total_bounds
        if not np.any(np.isnan(total_bounds)):
            xmin, ymin, xmax, ymax = total_bounds
            
            # Add buffer
            xmin -= buffer
            ymin -= buffer
            xmax += buffer
            ymax += buffer
            
            return (xmin, ymin, xmax, ymax)
        
        # Fallback bounds
        return (-1000, -1000, 1000, 1000)
    
    def visualize_map(self, save_path=None, figsize=(20, 20), dpi=300, show=True,
                     region=None, highlight_lanes=None, style='dark_background'):
        """
        Visualize the map data.
        
        Args:
            save_path (str, optional): Path to save the visualization image. If None, the image is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches.
            dpi (int, optional): DPI for the output image.
            show (bool, optional): Whether to display the plot.
            region (tuple, optional): Region to display as (xmin, ymin, xmax, ymax). If None, shows the entire map.
            highlight_lanes (list, optional): List of lane IDs to highlight.
            style (str, optional): Matplotlib style to use.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Set style
        plt.style.use(style)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Add lanes if available
        if self.lane_gdf is not None and not self.lane_gdf.empty:
            # Set up color mapping for lane types
            unique_types = self.lane_gdf['type'].unique()
            color_list = list(mcolors.TABLEAU_COLORS.values())
            # Ensure we have enough colors
            while len(color_list) < len(unique_types):
                color_list.extend(list(mcolors.CSS4_COLORS.values())[:len(unique_types) - len(color_list)])
            
            type_color_map = dict(zip(unique_types, color_list[:len(unique_types)]))
            
            # Get lanes to highlight, if any
            highlight_set = set(highlight_lanes) if highlight_lanes else set()
            
            # Plot each lane with its type color
            for idx, row in self.lane_gdf.iterrows():
                color = type_color_map.get(row['type'], 'gray')
                linewidth = 2 if row['id'] in highlight_set else 1
                alpha = 1.0 if row['id'] in highlight_set else 0.7
                ax.plot(*row.geometry.xy, color=color, linewidth=linewidth, alpha=alpha)
                
                # Add lane ID text for highlighted lanes
                if row['id'] in highlight_set:
                    mid_point = row.geometry.interpolate(0.5, normalized=True)
                    ax.text(mid_point.x, mid_point.y, row['id'], fontsize=8, 
                           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # Add lane type legend
            handles = [plt.Line2D([0], [0], color=color, linewidth=2) for color in type_color_map.values()]
            labels = list(type_color_map.keys())
            ax.legend(handles, labels, title="Lane Types", loc="upper right")
        else:
            ax.text(0.5, 0.5, "No lane data available to visualize", 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
        
        # Set title and axis labels
        ax.set_title(f"Map Visualization: {self.map_id}", fontsize=16)
        ax.set_xlabel("X Coordinate (m)", fontsize=12)
        ax.set_ylabel("Y Coordinate (m)", fontsize=12)
        
        # Set the view region
        if region:
            ax.set_xlim(region[0], region[2])
            ax.set_ylim(region[1], region[3])
        else:
            bounds = self.get_map_bounds()
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add a scale bar (100m)
        self._add_scale_bar(ax, 100)
        
        # Add map stats if we have lanes
        if self.lane_gdf is not None and not self.lane_gdf.empty:
            total_length = self.lane_gdf['length'].sum()
            stats_text = (
                f"Total lane features: {len(self.lane_gdf)}\n"
                f"Total length: {total_length:.1f}m\n"
                f"Avg feature length: {self.lane_gdf['length'].mean():.1f}m\n"
                f"Total points: {self.lane_gdf['num_points'].sum()}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Map visualization saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def _add_scale_bar(self, ax, length=100):
        """Add a scale bar to the plot."""
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # Place scale bar in bottom right
        bar_x = xmax - length - (xmax-xmin)*0.05
        bar_y = ymin + (ymax-ymin)*0.05
        
        # Draw the scale bar
        ax.plot([bar_x, bar_x + length], [bar_y, bar_y], 'k-', linewidth=2)
        
        # Add text
        ax.text(bar_x + length/2, bar_y + (ymax-ymin)*0.01, f"{length}m", 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, pad=2))


def visualize_all_maps(maps_folder, output_folder=None, style='dark_background'):
    """
    Visualize all map files in a folder.
    
    Args:
        maps_folder (str): Path to the folder containing map JSON files.
        output_folder (str, optional): Path to save visualizations. If None, creates a 'map_visualizations' folder.
        style (str): Matplotlib style to use.
    """
    # Create output folder if it doesn't exist
    if output_folder is None:
        output_folder = "map_visualizations"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all JSON files in the maps folder
    map_files = [f for f in os.listdir(maps_folder) if f.endswith('.json')]
    
    for map_file in map_files:
        map_path = os.path.join(maps_folder, map_file)
        map_id = os.path.splitext(map_file)[0]
        
        print(f"Visualizing map: {map_id}")
        
        try:
            # Create visualizer and visualize
            visualizer = MapVisualizer(map_path)
            save_path = os.path.join(output_folder, f"{map_id}.png")
            visualizer.visualize_map(save_path=save_path, show=False, style=style)
        except Exception as e:
            print(f"Error visualizing map {map_id}: {e}")


if __name__ == "__main__":
    # Example usage
    maps_folder = "maps"
    
    # Visualize a specific map
    map_path = "maps/0a2a67ef-0680-48d8-8d6c-6012a477b4bc.json"
    visualizer = MapVisualizer(map_path)
    
    # Create a full map visualization
    visualizer.visualize_map(save_path="map_visualization_full.png", show=True, style='dark_background')
    
    # Visualize all maps if needed
    # visualize_all_maps(maps_folder) 