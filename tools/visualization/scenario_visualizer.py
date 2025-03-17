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
import math
import matplotlib.transforms as mtransforms

def load_token_data(token_path):
    """Load token data from JSON file."""
    with open(token_path, 'r') as f:
        return json.load(f)

def load_map_data(map_id, maps_dir="maps"):
    """Load map data based on map_id."""
    map_path = os.path.join(maps_dir, f"{map_id}.json")
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            return json.load(f)
    print(f"Warning: Map file {map_path} not found")
    return None

def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw angle (rotation around z axis)."""
    # Convert quaternion to Euler angles (yaw)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

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
        'poses': token_data['EgoVehicleInformation'].get('historical_poses', []),
        'color': 'skyblue'
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
            'poses': token_data['TvInformation'].get('historical_poses', []),
            'color': 'salmon'
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
            'poses': veh.get('historical_poses', []),
            'color': 'lightgreen'
        }
        vehicles.append(other)
    
    return vehicles

def transform_map_coordinates(map_data, ego_pose=None):
    """Transform map coordinates to match the global coordinate system and ego vehicle orientation."""
    if not map_data or 'coordinate_origin' not in map_data:
        return 0, 0, 0
    
    # Get map origin from coordinate_origin
    origin_x = map_data['coordinate_origin']['translation']['x']
    origin_y = map_data['coordinate_origin']['translation']['y']
    origin_z = map_data['coordinate_origin']['translation']['z']
    
    # Extract rotation matrix from coordinate_origin
    rotation_data = map_data['coordinate_origin']['rotation']['data']
    rotation_matrix = np.array(rotation_data).reshape(3, 3)
    
    # Default rotation angle (no rotation)
    rotation_angle = 0
    
    # If ego pose has orientation info, use it to rotate the map
    if ego_pose and 'yaw' in ego_pose:
        rotation_angle = ego_pose['yaw']
    
    return origin_x, origin_y, rotation_angle

def visualize_map(ax, map_data, offset_x=0, offset_y=0, rotation_angle=0):
    """Visualize map data on the given axis with rotation."""
    if not map_data:
        return
    
    # Calculate rotation matrices
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    
    # Draw lanes
    for lane in map_data.get('lanes', []):
        if lane.get('global_lane_id', -1) < 0:
            continue  # Skip invalid lanes
        
        # Draw center line if available (as dashed gray line)
        center_line = lane.get('center_line', [])
        if center_line and len(center_line) > 1:
            # Transform coordinates with rotation and translation
            x_coords = []
            y_coords = []
            for pt in center_line:
                # Get original coordinates
                x_orig = pt.get('x', 0)
                y_orig = pt.get('y', 0)
                
                # Apply rotation
                x_rot = x_orig * cos_angle - y_orig * sin_angle
                y_rot = x_orig * sin_angle + y_orig * cos_angle
                
                # Apply translation
                x_coords.append(x_rot + offset_x)
                y_coords.append(y_rot + offset_y)
                
            ax.plot(x_coords, y_coords, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
        
        # Prepare arrays to collect left and right boundary points
        left_boundary_x = []
        left_boundary_y = []
        right_boundary_x = []
        right_boundary_y = []
        
        # Process lane segments and collect boundary points
        for segment_idx, segment in enumerate(lane.get('lane_segments', [])):
            # The map format seems to have lane_segments with left_boundary and right_boundary
            if 'left_boundary' in segment and 'right_boundary' in segment:
                # Try to get corresponding point for this segment
                if segment_idx < len(center_line):
                    # Get original centerline coordinates
                    x_orig = center_line[segment_idx].get('x', 0)
                    y_orig = center_line[segment_idx].get('y', 0)
                    
                    # Apply rotation
                    x_rot = x_orig * cos_angle - y_orig * sin_angle
                    y_rot = x_orig * sin_angle + y_orig * cos_angle
                    
                    # Apply translation
                    segment_x = x_rot + offset_x
                    segment_y = y_rot + offset_y
                    
                    # Get boundary offsets
                    left_offset = segment['left_boundary'].get('offset_to_center_line', 1.8)
                    right_offset = segment['right_boundary'].get('offset_to_center_line', 1.8)
                    
                    # Apply rotation to offsets - rotate offset direction by the same angle
                    # For left boundary (typically offset in +y direction)
                    left_x_offset = -left_offset * sin_angle
                    left_y_offset = left_offset * cos_angle
                    
                    # For right boundary (typically offset in -y direction)
                    right_x_offset = right_offset * sin_angle
                    right_y_offset = -right_offset * cos_angle
                    
                    # Add boundary points
                    left_boundary_x.append(segment_x + left_x_offset)
                    left_boundary_y.append(segment_y + left_y_offset)
                    
                    right_boundary_x.append(segment_x + right_x_offset)
                    right_boundary_y.append(segment_y + right_y_offset)
        
        # Draw left and right boundaries as solid black lines if we have points
        if left_boundary_x:
            ax.plot(left_boundary_x, left_boundary_y, color='black', 
                    linestyle='solid', linewidth=1.0, alpha=0.8, zorder=2)
        
        if right_boundary_x:
            ax.plot(right_boundary_x, right_boundary_y, color='black', 
                    linestyle='solid', linewidth=1.0, alpha=0.8, zorder=2)

def visualize_scenario(token_path, output_dir, create_video=True, view_radius=None, maps_dir="maps", output_format="gif"):
    """Visualize a scenario from a token file."""
    token_data = load_token_data(token_path)
    vehicles = extract_vehicle_data(token_data)
    token_name = token_path.split("/")[-1]
    # Get map ID from token and load the map
    map_id = token_data.get('MapId')
    map_data = None
    if map_id:
        map_data = load_map_data(map_id, maps_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the maximum number of timestamps (should be 30)
    max_timestamps = 30
    
    # Get initial ego vehicle position to fix the center of the plot
    initial_center_x = None
    initial_center_y = None
    if len(vehicles) > 0 and len(vehicles[0]['trajectories']) > 0:
        initial_center_x = vehicles[0]['trajectories'][0]['x']
        initial_center_y = vehicles[0]['trajectories'][0]['y']
        pose = vehicles[0]['poses'][0]
        initial_yaw = quaternion_to_yaw(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
    
    # Extract boundaries for consistent plot dimensions
    all_x = []
    all_y = []
    for veh in vehicles:
        for traj in veh['trajectories']:
            all_x.append(traj['x'])
            all_y.append(traj['y'])
    
    # Set fixed view radius to show a 200x200 meter window (100 meters in each direction)
    view_radius = 80.0
    
    # Set fixed center coordinates based on initial ego vehicle position
    # Use initial ego position as center, with fallback to predefined values if no ego data
    fixed_center_x = initial_center_x + np.cos(initial_yaw) * (view_radius - 15) if initial_center_x is not None else 25850
    fixed_center_y = initial_center_y + np.sin(initial_yaw) * (view_radius - 15) if initial_center_y is not None else -3950
    
    # Calculate plot boundaries once with the fixed center
    plot_min_x = fixed_center_x - view_radius
    plot_max_x = fixed_center_x + view_radius
    plot_min_y = fixed_center_y - view_radius
    plot_max_y = fixed_center_y + view_radius
    
    # Create frames for each timestamp
    frames = []
    
    for t_idx in tqdm(range(max_timestamps), desc="Creating frames"):
        # Create a square figure with fixed size
        fig, ax = plt.subplots(figsize=(12, 12))  # Square figure with fixed dimensions
        
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        # Ensure equal scaling for correct vehicle proportions
        ax.set_aspect('equal')
        plt.axis('equal')  # Additional enforcement of equal scale
        
        # Get current ego position and orientation for this frame (for map transformation)
        current_ego_pose = None
        if len(vehicles) > 0:
            # First, get initial orientation of ego vehicle
            initial_yaw = 0
            if (len(vehicles[0]['poses']) > 0 and 
                isinstance(vehicles[0]['poses'][0], dict) and 
                all(k in vehicles[0]['poses'][0] for k in ['qx', 'qy', 'qz', 'qw'])):
                # Extract yaw from quaternion
                pose = vehicles[0]['poses'][0]
                initial_yaw = quaternion_to_yaw(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
            
            # Then get current position for this frame
            if t_idx < len(vehicles[0]['trajectories']):
                current_ego_pose = {
                    'x': vehicles[0]['trajectories'][t_idx]['x'],
                    'y': vehicles[0]['trajectories'][t_idx]['y'],
                    'yaw': initial_yaw  # Use initial yaw for map rotation
                }
        
        # Visualize map before vehicles so vehicles are drawn on top
        if map_data:
            offset_x, offset_y, rotation_angle = transform_map_coordinates(map_data, current_ego_pose)
            visualize_map(ax, map_data, offset_x, offset_y, rotation_angle)
        
        # Plot each vehicle that has data for this timestamp
        for veh in vehicles:
            if t_idx < len(veh['timestamps']) and t_idx < len(veh['trajectories']):
                # Plot the vehicle position
                x = veh['trajectories'][t_idx]['x']
                y = veh['trajectories'][t_idx]['y']
                
                # Draw the vehicle as a rectangle
                length = veh['box']['length']
                width = veh['box']['width']
                
                # Get orientation from pose if available
                rotation_angle = 0
                if 'poses' in veh and t_idx < len(veh['poses']):
                    pose = veh['poses'][t_idx]
                    if isinstance(pose, dict) and all(k in pose for k in ['qx', 'qy', 'qz', 'qw']):
                        yaw = quaternion_to_yaw(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
                        rotation_angle = np.degrees(yaw)  # Convert to degrees for matplotlib
                    elif len(veh['poses']) > 1 and t_idx > 0:
                        # If no pose data, estimate orientation from movement direction
                        prev_x = veh['trajectories'][t_idx-1]['x']
                        prev_y = veh['trajectories'][t_idx-1]['y']
                        if abs(x - prev_x) > 0.001 or abs(y - prev_y) > 0.001:  # Only if moved enough
                            rotation_angle = np.degrees(np.arctan2(y - prev_y, x - prev_x))
                
                # Create vehicle polygon for proper rotation and dimensions
                corners_x = [-length/2, length/2, length/2, -length/2, -length/2]
                corners_y = [-width/2, -width/2, width/2, width/2, -width/2]
                
                # Rotate corners
                cos_angle = np.cos(np.radians(rotation_angle))
                sin_angle = np.sin(np.radians(rotation_angle))
                
                rotated_x = []
                rotated_y = []
                for corner_x, corner_y in zip(corners_x, corners_y):
                    # Rotate point around origin
                    rx = corner_x * cos_angle - corner_y * sin_angle
                    ry = corner_x * sin_angle + corner_y * cos_angle
                    
                    # Translate to vehicle position
                    rotated_x.append(x + rx)
                    rotated_y.append(y + ry)
                
                # First draw a white background polygon to fully cover map lines
                background_polygon = plt.Polygon(np.column_stack([rotated_x, rotated_y]), 
                                             closed=True, 
                                             edgecolor='none',
                                             facecolor='white',
                                             alpha=1.0,  # Fully opaque
                                             zorder=9)   # Above map but below vehicle color
                ax.add_patch(background_polygon)
                
                # Draw the colored vehicle polygon on top
                vehicle_polygon = plt.Polygon(np.column_stack([rotated_x, rotated_y]), 
                                             closed=True, 
                                             edgecolor='black',
                                             facecolor=veh['color'],
                                             alpha=1.0,   # Fully opaque 
                                             linewidth=1.5,
                                             zorder=10)  # Higher zorder to be on top of map
                ax.add_patch(vehicle_polygon)
                
                # Add a directional arrow to show orientation
                arrow_length = length * 0.6
                dx = arrow_length * np.cos(np.radians(rotation_angle))
                dy = arrow_length * np.sin(np.radians(rotation_angle))
                # Draw a direction line instead of an arrow
                ax.plot([x, x + dx], [y, y + dy], color='black', linewidth=2, solid_capstyle='round', zorder=11)

                # If this is the ego vehicle, add a highlight
                if veh['id'] == 'ego':
                    ego_circle = plt.Circle((x, y), radius=max(length, width)*0.7, 
                                          fill=False, color='cyan', linestyle='-', linewidth=2, alpha=0.5,
                                          zorder=9)  # Just below vehicles but above map
                    ax.add_patch(ego_circle)
        
        # Add map information to the display
        if map_data:
            map_info = f"Map ID: {map_id}\n"
            map_info += f"Lanes: {map_data.get('total_lanes_in_hdmap', 'Unknown')}"
            
            # Add the map info text in the bottom left corner
            ax.text(0.02, 0.02, map_info, transform=ax.transAxes, fontsize=12,
                   verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add timestamp and frame information
        moment_id = token_data.get('MomentId', 'Unknown')
        timestamp = token_data['Timestamp'] if t_idx == max_timestamps-1 else None
        
        title = f"Scenario: {moment_id}\n"
        title += f"Frame {t_idx+1}/{max_timestamps}"
        if timestamp:
            title += f" - Timestamp: {timestamp}"
        
        ax.set_title(title, fontsize=14)
        
        # Add scene information
        scene_info = token_data.get('SceneInformation', {})
        info_text = f"Weather: {scene_info.get('weather_conditions', 'Unknown')}\n"
        info_text += f"Time: {scene_info.get('time_of_day', 'Unknown')}\n"
        info_text += f"Speed Limit: {scene_info.get('traffic_speed_limit', 'Unknown')} m/s\n"
        info_text += f"Vehicles in Scope: {scene_info.get('total_vehicles_in_scope', 'Unknown')}"
        
        # Add the text in the corner
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add a scale bar
        scale_bar_length = 10  # meters
        scale_x_start = plot_max_x - scale_bar_length - 5
        scale_y = plot_min_y + 5
        ax.plot([scale_x_start, scale_x_start + scale_bar_length], [scale_y, scale_y], 'k-', lw=2)
        ax.text(scale_x_start + scale_bar_length/2, scale_y + 2, f"{scale_bar_length}m", 
                ha='center', va='bottom', fontsize=10)
        
        # Add grid for better visualization
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        
        # Save the frame
        frame_path = os.path.join(output_dir, f"frame_{t_idx:03d}.png")
        
        # Save with fixed dimensions, without tight bbox adjustment
        plt.savefig(frame_path, dpi=120, bbox_inches=None)
        frames.append(frame_path)
        plt.close(fig)
    
    if create_video:
        # Create a video from frames
        video_path = os.path.join(output_dir, f"{token_name}.{output_format}")
        create_video_from_frames(frames, video_path, token_name, output_format=output_format)
        
        print(f"Video saved to: {video_path}")
    
    return frames

def create_video_from_frames(frame_paths, output_path, token_name, fps=10, output_format="gif"):
    """Create a video from a list of frame paths."""
    if not frame_paths:
        print("No frames to create video")
        return
    
    # Ensure output path has the correct extension
    if not output_path.endswith(f'.{output_format}'):
        output_path = os.path.splitext(output_path)[0] + f'.{output_format}'
    
    if output_format.lower() == "gif":
        try:
            import imageio
            
            # Read all frames
            frames = []
            for frame_path in tqdm(frame_paths, desc="Reading frames for GIF"):
                frames.append(imageio.imread(frame_path))
            
            # Save as GIF
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"GIF created successfully: {output_path}")
        except ImportError:
            print("Creating GIF requires imageio library. Please install it with: pip install imageio")
            print("Falling back to MP4 video creation...")
            create_mp4_video(frame_paths, output_path.replace('.gif', '.mp4'), fps)
    else:  # Default to MP4
        create_mp4_video(frame_paths, output_path, fps)

def create_mp4_video(frame_paths, output_path, fps=10):
    """Create an MP4 video from a list of frame paths using OpenCV."""
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
    parser.add_argument('--follow_ego', action='store_true',
                      help='Center visualization on ego vehicle')
    parser.add_argument('--view_radius', type=float, default=50.0,
                      help='Radius around ego vehicle to display (in meters)')
    parser.add_argument('--maps_dir', type=str, default='maps',
                      help='Directory containing map files')
    parser.add_argument('--output_format', type=str, choices=['gif', 'mp4'], default='mp4',
                      help='Output format for the video (gif or mp4)')
    
    args = parser.parse_args()
    
    # Only use view_radius if follow_ego is enabled
    view_radius = args.view_radius if args.follow_ego else None
    
    visualize_scenario(args.token_path, args.output_dir, not args.no_video, view_radius, args.maps_dir, args.output_format)

if __name__ == "__main__":
    main() 