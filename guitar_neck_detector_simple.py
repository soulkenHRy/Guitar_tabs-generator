#!/usr/bin/env python3
"""
Simple Guitar Neck Detector
Detects guitar necks in any video file using the same technique as guitar_detector_gui.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import time


# Guitar Neck Detection Configuration
MODEL_PATH = "runs/neck_nut_detection/train2/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class names - NEW MODEL with neck and nut detection
CLASS_NAMES = ['neck', 'nut']

# Colors for each class (BGR format)
COLORS = {
    'neck': (0, 255, 255),            # Yellow
    'nut': (255, 0, 255),             # Magenta
}

# Fret positions as percentage of fretboard length
FRET_POSITIONS = {
    0: 0.0, 1: 8.17, 2: 15.88, 3: 23.18, 4: 30.10, 5: 36.66,
    6: 42.89, 7: 48.83, 8: 54.48, 9: 59.86, 10: 65.00, 11: 69.92,
    12: 72.75, 13: 76.84, 14: 80.70, 15: 84.35, 16: 87.80,
    17: 91.08, 18: 94.20, 19: 97.16, 20: 100.0,
}


def draw_frets_on_neck(frame, x1, y1, x2, y2, num_frets=20):
    """Draw fret lines on the detected guitar neck region (same as GUI)."""
    neck_width = x2 - x1
    
    # Draw fret bars
    for fret_num in range(num_frets + 1):
        if fret_num not in FRET_POSITIONS:
            continue
        
        fret_percentage = (100.0 - FRET_POSITIONS[fret_num]) / 100.0
        fret_x = int(x1 + (neck_width * fret_percentage))
        
        # Style the bars
        if fret_num == 0:
            color, thickness = (255, 255, 255), 3  # Nut
        elif fret_num == 12:
            color, thickness = (0, 255, 255), 2  # 12th fret marker
        else:
            color, thickness = (200, 200, 200), 1
        
        cv2.line(frame, (fret_x, y1), (fret_x, y2), color, thickness)
    
    # Label the SPACES (frets) between bars
    for fret_num in range(1, num_frets + 1):
        if fret_num not in FRET_POSITIONS or (fret_num - 1) not in FRET_POSITIONS:
            continue
        
        # Get positions of the two bars that define this fret space
        prev_bar_percentage = (100.0 - FRET_POSITIONS[fret_num - 1]) / 100.0
        curr_bar_percentage = (100.0 - FRET_POSITIONS[fret_num]) / 100.0
        
        # Calculate middle of the space
        middle_percentage = (prev_bar_percentage + curr_bar_percentage) / 2.0
        label_x = int(x1 + (neck_width * middle_percentage))
        
        # Draw fret number in the middle of the space
        label = str(fret_num)
        # Add special highlighting for important frets
        if fret_num in [1, 3, 5, 7, 9, 12, 15, 17, 19]:
            color = (255, 255, 0)  # Yellow for marker frets
            thickness = 2
        else:
            color = (200, 200, 200)  # Light gray
            thickness = 1
        
        cv2.putText(frame, label, (label_x - 8, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


def detect_guitar_neck(video_path, output_path=None, display=True):
    """
    Detect guitar necks in a video file using the same technique as guitar_detector_gui.py
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        display: Whether to display the video while processing
    """
    
    # Load YOLO model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Processing stats
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
    # Temporal smoothing for nut detection
    last_nut_position = None
    nut_detection_history = []
    max_history = 5  # Keep last 5 detections
    
    print("\nProcessing video... Press 'q' to quit")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame at 640x480 for detection (same as GUI)
        process_frame = cv2.resize(frame, (640, 480))
        
        # Run YOLO detection with lower confidence for nut
        results = model.predict(process_frame, conf=0.3,  # Lower threshold for better nut detection
                               iou=IOU_THRESHOLD, imgsz=416, verbose=False)
        
        # Calculate scale factors to map back to original frame
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 480
        
        # Track if nut was detected this frame
        nut_detected_this_frame = False
        current_nut_position = None
        
        # Process detections
        detections = results[0].boxes
        for box in detections:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Scale back to original frame size
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Get class name
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            color = COLORS.get(class_name, (0, 255, 0))
            
            # Process both neck and nut classes
            if class_name == 'neck' and conf >= CONFIDENCE_THRESHOLD:
                detection_count += 1
                
                # Draw frets on neck
                draw_frets_on_neck(frame, x1, y1, x2, y2)
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif class_name == 'nut' and conf >= 0.25:  # Even lower threshold for nut
                nut_detected_this_frame = True
                current_nut_position = (x1, y1, x2, y2, conf)
                detection_count += 1
            elif class_name == 'nut' and conf >= 0.25:  # Even lower threshold for nut
                nut_detected_this_frame = True
                current_nut_position = (x1, y1, x2, y2, conf)
                detection_count += 1
        
        # Temporal smoothing: use last known position if no detection this frame
        if nut_detected_this_frame:
            # Add to history
            nut_detection_history.append(current_nut_position)
            if len(nut_detection_history) > max_history:
                nut_detection_history.pop(0)
            last_nut_position = current_nut_position
        elif last_nut_position is not None and len(nut_detection_history) > 0:
            # Use smoothed position from history if available
            current_nut_position = last_nut_position
        
        # Draw nut if we have a position (detected or from history)
        if current_nut_position is not None:
            x1, y1, x2, y2, conf = current_nut_position
            color = COLORS.get('nut', (255, 0, 255))
            
            # Draw the nut bounding box with different style
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            status = "LIVE" if nut_detected_this_frame else "TRACK"
            cv2.putText(frame, f"NUT [{status}]: {conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate nut center
            nut_center_x = (x1 + x2) // 2
            nut_center_y = (y1 + y2) // 2
            
            # Calculate angle of the nut (assuming width > height means horizontal orientation)
            nut_width = x2 - x1
            nut_height = y2 - y1
            
            # Determine the angle based on the nut's orientation
            # If the nut is wider than tall, draw horizontal line
            # If taller than wide, draw vertical line
            if nut_width > nut_height:
                # Horizontal nut - draw line along width
                angle_rad = 0  # Horizontal
                line_length = nut_width // 2
            else:
                # Vertical nut - draw line along height
                angle_rad = np.pi / 2  # Vertical
                line_length = nut_height // 2
            
            # Calculate line endpoints
            line_x1 = int(nut_center_x - line_length * np.cos(angle_rad))
            line_y1 = int(nut_center_y - line_length * np.sin(angle_rad))
            line_x2 = int(nut_center_x + line_length * np.cos(angle_rad))
            line_y2 = int(nut_center_y + line_length * np.sin(angle_rad))
            
            # Draw the line through the nut
            cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (0, 255, 0), 4)
            cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (255, 255, 255), 2)
            
            # Draw a circle at nut center
            cv2.circle(frame, (nut_center_x, nut_center_y), 8, color, -1)
            cv2.circle(frame, (nut_center_x, nut_center_y), 8, (255, 255, 255), 2)
        
        # FPS calculation
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Add frame info (same style as GUI)
        cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        
        # Write frame to output video
        if out:
            out.write(frame)
        
        # Display frame
        if display:
            # Resize for display if too large
            display_frame = frame
            if width > 1280:
                scale = 1280 / width
                new_width = 1280
                new_height = int(height * scale)
                display_frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow('Guitar Neck Detection', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f} | Detections: {detection_count}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total neck detections: {detection_count}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed:.2f}")
    if output_path:
        print(f"Output saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Simple Guitar Neck Detector - Detect guitar necks in any video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect and display only (no output file)
  python guitar_neck_detector_simple.py input_video.mp4
  
  # Detect and save to output file
  python guitar_neck_detector_simple.py input_video.mp4 -o output_video.mp4
  
  # Process without display (headless mode)
  python guitar_neck_detector_simple.py input_video.mp4 -o output.mp4 --no-display
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='Path to output video file (optional)')
    parser.add_argument('--no-display', action='store_true', 
                        help='Run without displaying video (headless mode)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Update confidence threshold if provided
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence
    
    # Run detection
    detect_guitar_neck(
        video_path=video_path,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
