import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import glob
import imageio

def create_video_grid_animation(input_folder, output_gif_path, grid_size=(7, 7), fps=10, 
                               optimize_size=True, resize_factor=1.0,
                               center_duration=3, full_grid_duration=3, transition_duration=8,
                               output_resolution=(1280, 1280)):
    """
    Create a 5x5 grid of videos with zoom animation and save as GIF.
    
    Args:
        input_folder: Folder containing video files
        output_gif_path: Path to save the output GIF
        grid_size: Tuple (rows, cols) for the grid layout
        fps: Frames per second for the output GIF
        optimize_size: Whether to optimize GIF size
        resize_factor: Factor to resize videos (0.5 = half size)
        center_duration: Time in seconds to show center video
        full_grid_duration: Time in seconds to show full grid
        transition_duration: Time in seconds for zoom transitions
        output_resolution: Final resolution of the GIF (width, height)
    """
    # Get list of video files in the folder
    video_files = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    
    if len(video_files) < grid_size[0] * grid_size[1]:
        raise ValueError(f"Need {grid_size[0] * grid_size[1]} videos but only found {len(video_files)}")
    
    # Take the first 25 videos if there are more
    video_files = video_files[:grid_size[0] * grid_size[1]]
    
    # Open all videos
    video_captures = [cv2.VideoCapture(file) for file in video_files]
    
    # Get video dimensions (assuming all videos have the same dimensions)
    _, first_frame = video_captures[0].read()
    if first_frame is None:
        raise ValueError("Could not read the first frame of the first video")
    
    # Apply resize factor if needed
    frame_height, frame_width = first_frame.shape[:2]
    if resize_factor != 1.0:
        frame_height = int(frame_height * resize_factor)
        frame_width = int(frame_width * resize_factor)
    
    # Calculate grid dimensions
    grid_width = frame_width * grid_size[1]
    grid_height = frame_height * grid_size[0]
    
    # Reset all videos to the beginning
    for cap in video_captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Define animation phases and durations
    total_duration = center_duration + transition_duration + full_grid_duration + transition_duration
    
    # Number of frames for each phase
    center_frames = int(center_duration * fps)
    zoom_out_frames = int(transition_duration * fps)
    full_grid_frames = int(full_grid_duration * fps)
    zoom_in_frames = int(transition_duration * fps)
    
    # Total frames for the animation
    total_frames = center_frames + zoom_out_frames + full_grid_frames + zoom_in_frames
    
    # Create output frames list
    frames_for_gif = []
    
    # Center video position
    center_row, center_col = grid_size[0] // 2, grid_size[1] // 2
    center_x = center_col * frame_width
    center_y = center_row * frame_height
    
    # Zoom levels
    single_zoom = 1.0  # Just the center video
    full_zoom = grid_size[0]  # Full grid
    
    # Generate frames for each phase of the animation
    for frame_idx in range(total_frames):
        # Determine current phase and calculate zoom level
        if frame_idx < center_frames:
            # Phase 1: Zoom in on center video and stay there for 3 seconds
            # Start slightly zoomed out and gradually zoom in to single video
            progress = min(1.0, frame_idx / (center_frames/3))  # Complete zoom in the first third of this phase
            # Start at 2x zoom and go to 1x (full zoom on center)
            current_zoom = full_zoom#2.0 - progress
            
        elif frame_idx < center_frames + zoom_out_frames:
            # Phase 2: Slowly zoom out to show all videos
            progress = (frame_idx - center_frames) / zoom_out_frames
            current_zoom = full_zoom - progress * (full_zoom - single_zoom)
            
        elif frame_idx < center_frames + zoom_out_frames + full_grid_frames:
            # Phase 3: Stay at full grid view for 3 seconds
            current_zoom = single_zoom
            
        else:
            # Phase 4: Zoom back to center
            progress = (frame_idx - (center_frames + zoom_out_frames + full_grid_frames)) / zoom_in_frames
            current_zoom = single_zoom + progress * (full_zoom - single_zoom)
        
        # Calculate visible area based on zoom
        visible_width = grid_width / current_zoom
        visible_height = grid_height / current_zoom
        
        # Calculate top-left corner of visible area
        visible_x = center_x + frame_width/2 - visible_width/2
        visible_y = center_y + frame_height/2 - visible_height/2
        
        # Create a blank grid image with white background instead of black
        # Using 255 for all BGR channels to ensure pure white
        grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Read current frame from each video and place in grid
        for i, cap in enumerate(video_captures):
            # Reset video if it's at the end
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            ret, video_frame = cap.read()
            if not ret:
                # If failed to read, use a black frame
                video_frame = np.zeros((int(frame_height/resize_factor), int(frame_width/resize_factor), 3), dtype=np.uint8)
            
            # Resize frame if needed
            if resize_factor != 1.0:
                video_frame = cv2.resize(video_frame, (frame_width, frame_height))
            
            # Calculate position in grid
            row = i // grid_size[1]
            col = i % grid_size[1]
            
            y_start = row * frame_height
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width
            
            # Place video frame in grid
            grid_image[y_start:y_end, x_start:x_end] = video_frame
        
        # Crop to visible area and resize to maintain consistent output size
        visible_area = grid_image[int(visible_y):int(visible_y + visible_height), 
                                  int(visible_x):int(visible_x + visible_width)]
        
        # Resize to output dimensions
        output_frame = cv2.resize(visible_area, (grid_width, grid_height))
        
        # Resize to the requested fixed resolution
        output_frame = cv2.resize(output_frame, output_resolution)
        
        # Convert from BGR to RGB for PIL
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        
        # Ensure pure white background by thresholding near-white colors
        # This fixes any color shifts from interpolation during resizing
        white_mask = (output_frame_rgb > 245).all(axis=2)
        output_frame_rgb[white_mask] = [255, 255, 255]
        
        pil_image = Image.fromarray(output_frame_rgb)
        
        frames_for_gif.append(pil_image)
    
    # Release video captures
    for cap in video_captures:
        cap.release()
    
    print(f"Creating GIF with {len(frames_for_gif)} frames...")
    
    # Save as GIF with optimization for smaller file size
    if optimize_size:
        # Use PIL for better optimization
        frames_for_gif[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames_for_gif[1:],
            optimize=True,
            duration=1000/fps,  # Duration in milliseconds between frames
            loop=0,  # Loop forever
            disposal=2,  # Clear previous frame before rendering next
            # Add quantization to reduce color palette
            colors=256  # Limit color palette to 256 colors
        )
    else:
        # Use imageio for high quality but larger size
        imageio.mimsave(output_gif_path, frames_for_gif, fps=fps)
    
    print(f"GIF saved to {output_gif_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a 5x5 video grid animation')
    parser.add_argument('input_folder', help='Folder containing video files')
    parser.add_argument('--output', default='video_grid.gif', help='Output GIF path')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for output GIF')
    parser.add_argument('--center-time', type=float, default=3, help='Time to show center video (seconds)')
    parser.add_argument('--grid-time', type=float, default=3, help='Time to show full grid (seconds)')
    parser.add_argument('--transition-time', type=float, default=8, help='Time for zoom transitions (seconds)')
    parser.add_argument('--resize', type=float, default=0.5, help='Resize factor for videos (smaller = smaller GIF)')
    parser.add_argument('--resolution', type=int, default=1280, help='Output resolution (square)')
    parser.add_argument('--no-optimize', action='store_false', dest='optimize', 
                        help='Disable GIF size optimization')
    
    args = parser.parse_args()
    
    create_video_grid_animation(
        args.input_folder, 
        args.output, 
        grid_size=(7, 7), 
        fps=args.fps,
        center_duration=args.center_time,
        full_grid_duration=args.grid_time,
        transition_duration=args.transition_time,
        optimize_size=args.optimize,
        resize_factor=args.resize,
        output_resolution=(args.resolution, args.resolution)
    ) 