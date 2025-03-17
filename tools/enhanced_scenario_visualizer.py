import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

def load_token_data(token_path):
    """Load token data from JSON file."""
    with open(token_path, 'r') as f:
        return json.load(f)

def extract_vehicle_data(token_data):
    """Extract relevant vehicle data from token."""
    vehicles = []
    
    # Add ego vehicle
    ego = {
        'id': 'ego',
        'type': token_data['EgoVehicleInformation']['vehicle_type'],
        'box': token_data['EgoVehicleInformation']['vehicle_box'],
        'timestamps': token_data['EgoVehicleInformation']['historical_timestamp'],
        'trajectories': token_data['EgoVehicleInformation']['historical_trajectories'],
        'color': 'blue'
    }
    vehicles.append(ego)
    
    # Add target vehicle if it exists
    if 'TvInformation' in token_data:
        tv = {
            'id': f"tv_{token_data['TvInformation']['vehicle_id']}",
            'type': token_data['TvInformation']['vehicle_type'],
            'box': token_data['TvInformation']['vehicle_box'],
            'timestamps': token_data['TvInformation']['historical_timestamp'],
            'trajectories': token_data['TvInformation']['historical_trajectories'],
            'color': 'red'
        }
        vehicles.append(tv)
    
    # Add other vehicles
    for idx, veh in enumerate(token_data.get('OtherVehiclesInformation', [])):
        other = {
            'id': f"veh_{veh['vehicle_id']}",
            'type': veh['vehicle_type'],
            'box': veh['vehicle_box'],
            'timestamps': veh['historical_timestamp'],
            'trajectories': veh['historical_trajectories'],
            'color': 'green' if 'BARRIER' not in veh['vehicle_type'] else 'orange'
        }
        vehicles.append(other)
    
    return vehicles

def visualize_scenario(token_path, output_dir, create_video=True, show_trails=True):
    """Visualize a scenario from a token file."""
    token_data = load_token_data(token_path)
    vehicles = extract_vehicle_data(token_data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the maximum number of timestamps (should be 30)
    max_timestamps = 30
    
    # Extract boundaries for consistent plot dimensions
    all_x = []
    all_y = []
    for veh in vehicles:
        for traj in veh['trajectories']:
            all_x.append(traj['x'])
            all_y.append(traj['y'])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Add a margin to the boundaries
    margin = max(max_x - min_x, max_y - min_y) * 0.1
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    # Get scene information
    scene_info = token_data.get('SceneInformation', {})
    semantic_infra = token_data.get('SemanticInfrastructure', {})
    
    # Create frames for each timestamp
    frames = []
    for t_idx in tqdm(range(max_timestamps), desc="Creating frames"):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set plot boundaries for consistency
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Plot each vehicle that has data for this timestamp
        for veh in vehicles:
            # Draw the vehicle's past trajectory (trail)
            if show_trails and t_idx > 0:
                valid_indices = min(t_idx, len(veh['trajectories']))
                traj_x = [veh['trajectories'][i]['x'] for i in range(valid_indices)]
                traj_y = [veh['trajectories'][i]['y'] for i in range(valid_indices)]
                ax.plot(traj_x, traj_y, '-', color=veh['color'], alpha=0.5, linewidth=1)
            
            # Draw the current position
            if t_idx < len(veh['timestamps']) and t_idx < len(veh['trajectories']):
                # Plot the vehicle position
                x = veh['trajectories'][t_idx]['x']
                y = veh['trajectories'][t_idx]['y']
                
                # Draw the vehicle as a rectangle
                length = veh['box']['length']
                width = veh['box']['width']
                
                # Create a rectangle representing the vehicle
                rect = patches.Rectangle(
                    (x - length/2, y - width/2), 
                    length, width, 
                    linewidth=1, 
                    edgecolor=veh['color'], 
                    facecolor=veh['color'],
                    alpha=0.7
                )
                ax.add_patch(rect)
                
                # Add vehicle ID and type
                ax.text(x, y, f"{veh['id']}\n{veh['type'][:4]}", 
                       fontsize=8, ha='center', va='center', color='white',
                       bbox=dict(facecolor=veh['color'], alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Add timestamp and frame information
        moment_id = token_data.get('MomentId', 'Unknown')
        timestamp = token_data['Timestamp'] if t_idx == max_timestamps-1 else None
        
        title = f"Scenario: {moment_id}\n"
        title += f"Frame {t_idx+1}/{max_timestamps}"
        if timestamp:
            title += f" - Timestamp: {timestamp}"
        
        ax.set_title(title)
        
        # Add scene information
        info_text = f"Weather: {scene_info.get('weather_conditions', 'Unknown')}\n"
        info_text += f"Time: {scene_info.get('time_of_day', 'Unknown')}\n"
        info_text += f"Speed Limit: {scene_info.get('traffic_speed_limit', 'Unknown')} m/s\n"
        info_text += f"Vehicles in Scope: {scene_info.get('total_vehicles_in_scope', 'Unknown')}"
        
        # Add the text in the corner
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid for better visualization
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal')
        
        # Save the frame
        frame_path = os.path.join(output_dir, f"frame_{t_idx:03d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=120)
        frames.append(frame_path)
        plt.close(fig)
    
    if create_video:
        # Create a video from frames
        video_path = os.path.join(output_dir, "scenario_visualization.mp4")
        create_video_from_frames(frames, video_path)
        
        print(f"Video saved to: {video_path}")
    
    return frames

def create_video_from_frames(frame_paths, output_path, fps=5):
    """Create a video from a list of frame paths."""
    if not frame_paths:
        print("No frames to create video")
        return
    
    # Read the first frame to get dimensions
    frame = cv2.imread(frame_paths[0])
    height, width, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each frame to the video
    for frame_path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video created successfully: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize vehicle scenario data')
    parser.add_argument('token_path', type=str, help='Path to the token JSON file')
    parser.add_argument('--output_dir', type=str, default='scenario_visualization',
                      help='Directory to save output visualizations')
    parser.add_argument('--no_video', action='store_true', 
                      help='Skip video creation and only produce image frames')
    parser.add_argument('--no_trails', action='store_true',
                      help='Hide vehicle trajectory trails')
    
    args = parser.parse_args()
    
    visualize_scenario(args.token_path, args.output_dir, not args.no_video, not args.no_trails)

if __name__ == "__main__":
    main() 