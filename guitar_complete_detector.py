#!/usr/bin/env python3
"""
Complete Guitar Detection System
Combines video-based fret detection with audio frequency analysis
Outputs both visual fret positions and audio frequency/fret mapping
"""

import cv2
import numpy as np
from ultralytics import YOLO
from cvzone.HandTrackingModule import HandDetector
import time
import argparse
import librosa
from pathlib import Path
from audio_preprocessor import AudioPreprocessor
import subprocess
import tempfile
import os


# Configuration
MODEL_PATH = "guitar_detector_best.pt"
CONFIDENCE_THRESHOLD = 0.5
NUT_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for nut detection
IOU_THRESHOLD = 0.45

# Class names - NEW MODEL with neck and nut detection
CLASS_NAMES = ['neck', 'nut']

# Colors for each class (BGR format)
COLORS = {
    'neck': (0, 255, 255),            # Yellow
    'nut': (255, 0, 255),             # Magenta
}

# Fret positions as percentage of fretboard length (from guitar_fretboard_datas)
FRET_POSITIONS = {
    0: 0.0,      # Nut
    1: 8.17,
    2: 15.88,
    3: 23.18,
    4: 30.10,
    5: 36.66,
    6: 42.89,
    7: 48.83,
    8: 54.48,
    9: 59.86,
    10: 65.00,
    11: 69.92,
    12: 72.75,   # 12th fret (octave)
    13: 76.84,
    14: 80.70,
    15: 84.35,
    16: 87.80,
    17: 91.08,
    18: 94.20,
    19: 97.16,
    20: 100.0,   # End of fretboard
}

# Guitar frequency mapping: (string, fret) -> frequency in Hz
GUITAR_FREQUENCIES = {
    # String 6 (Low E)
    (6, 0): 82.41, (6, 1): 87.31, (6, 2): 92.50, (6, 3): 98.00, (6, 4): 103.83,
    (6, 5): 110.00, (6, 6): 116.54, (6, 7): 123.47, (6, 8): 130.81, (6, 9): 138.59,
    (6, 10): 146.83, (6, 11): 155.56, (6, 12): 164.81, (6, 13): 174.61, (6, 14): 185.00,
    (6, 15): 196.00, (6, 16): 207.65, (6, 17): 220.00, (6, 18): 233.08, (6, 19): 246.94,
    (6, 20): 261.63,
    
    # String 5 (A)
    (5, 0): 110.00, (5, 1): 116.54, (5, 2): 123.47, (5, 3): 130.81, (5, 4): 138.59,
    (5, 5): 146.83, (5, 6): 155.56, (5, 7): 164.81, (5, 8): 174.61, (5, 9): 185.00,
    (5, 10): 196.00, (5, 11): 207.65, (5, 12): 220.00, (5, 13): 233.08, (5, 14): 246.94,
    (5, 15): 261.63, (5, 16): 277.18, (5, 17): 293.66, (5, 18): 311.13, (5, 19): 329.63,
    (5, 20): 349.23,
    
    # String 4 (D)
    (4, 0): 146.83, (4, 1): 155.56, (4, 2): 164.81, (4, 3): 174.61, (4, 4): 185.00,
    (4, 5): 196.00, (4, 6): 207.65, (4, 7): 220.00, (4, 8): 233.08, (4, 9): 246.94,
    (4, 10): 261.63, (4, 11): 277.18, (4, 12): 293.66, (4, 13): 311.13, (4, 14): 329.63,
    (4, 15): 349.23, (4, 16): 369.99, (4, 17): 392.00, (4, 18): 415.30, (4, 19): 440.00,
    (4, 20): 466.16,
    
    # String 3 (G)
    (3, 0): 196.00, (3, 1): 207.65, (3, 2): 220.00, (3, 3): 233.08, (3, 4): 246.94,
    (3, 5): 261.63, (3, 6): 277.18, (3, 7): 293.66, (3, 8): 311.13, (3, 9): 329.63,
    (3, 10): 349.23, (3, 11): 369.99, (3, 12): 392.00, (3, 13): 415.30, (3, 14): 440.00,
    (3, 15): 466.16, (3, 16): 493.88, (3, 17): 523.25, (3, 18): 554.37, (3, 19): 587.33,
    (3, 20): 622.25,
    
    # String 2 (B)
    (2, 0): 246.94, (2, 1): 261.63, (2, 2): 277.18, (2, 3): 293.66, (2, 4): 311.13,
    (2, 5): 329.63, (2, 6): 349.23, (2, 7): 369.99, (2, 8): 392.00, (2, 9): 415.30,
    (2, 10): 440.00, (2, 11): 466.16, (2, 12): 493.88, (2, 13): 523.25, (2, 14): 554.37,
    (2, 15): 587.33, (2, 16): 622.25, (2, 17): 659.25, (2, 18): 698.46, (2, 19): 739.99,
    (2, 20): 783.99,
    
    # String 1 (High E)
    (1, 0): 329.63, (1, 1): 349.23, (1, 2): 369.99, (1, 3): 392.00, (1, 4): 415.30,
    (1, 5): 440.00, (1, 6): 466.16, (1, 7): 493.88, (1, 8): 523.25, (1, 9): 554.37,
    (1, 10): 587.33, (1, 11): 622.25, (1, 12): 659.25, (1, 13): 698.46, (1, 14): 739.99,
    (1, 15): 783.99, (1, 16): 830.61, (1, 17): 880.00, (1, 18): 932.33, (1, 19): 987.77,
    (1, 20): 1046.50,
}


def find_closest_fret(frequency_hz, tolerance=2.0):
    """
    Find the closest guitar string and fret for a given frequency.
    
    Args:
        frequency_hz: Detected frequency in Hz
        tolerance: Maximum Hz difference to accept (default: 2.0)
    
    Returns:
        Tuple of (string, fret, exact_hz) or None if no match within tolerance
    """
    closest_match = None
    min_diff = float('inf')
    
    for (string, fret), target_hz in GUITAR_FREQUENCIES.items():
        diff = abs(frequency_hz - target_hz)
        if diff < min_diff:
            min_diff = diff
            closest_match = (string, fret, target_hz)
    
    # Only return if within tolerance
    if min_diff <= tolerance:
        return closest_match
    return None


def analyze_audio_segment(audio_segment, sr):
    """
    Analyze a segment of audio and detect frequency/fret.
    
    Args:
        audio_segment: Audio data array
        sr: Sample rate
    
    Returns:
        Tuple of (detected_hz, string, fret, exact_hz) or (None, None, None, None)
    """
    if len(audio_segment) < 512:  # Too short
        return None, None, None, None
    
    try:
        # Extract pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_segment, 
            sr=sr, 
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7')   # ~2093 Hz
        )
        
        # Clean up NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            detected_hz = np.median(f0_clean)
            
            # Find closest guitar fret
            match = find_closest_fret(detected_hz, tolerance=2.0)
            
            if match:
                string, fret, exact_hz = match
                return detected_hz, string, fret, exact_hz
    except Exception as e:
        pass
    
    return None, None, None, None


def get_frets_from_hand_position(hand_lmList, neck_x1, neck_y1, neck_x2, neck_y2, scale_x, scale_y):
    """
    Determine which frets a hand is covering based on landmark positions.
    
    Args:
        hand_lmList: List of hand landmarks from cvzone
        neck_x1, neck_y1, neck_x2, neck_y2: Neck bounding box coordinates
        scale_x, scale_y: Scaling factors for coordinate conversion
    
    Returns:
        Set of fret numbers that the hand is covering
    """
    covered_frets = set()
    neck_width = neck_x2 - neck_x1
    
    # Check each landmark of the hand
    for landmark in hand_lmList:
        # Scale landmark to original frame size
        lm_x = int(landmark[0] * scale_x)
        lm_y = int(landmark[1] * scale_y)
        
        # Check if landmark is within neck region
        if neck_x1 <= lm_x <= neck_x2 and neck_y1 <= lm_y <= neck_y2:
            # Calculate which fret region this landmark is in
            relative_x = lm_x - neck_x1
            relative_percentage = (relative_x / neck_width) * 100.0
            
            # Convert to actual fret position (remember: flipped orientation)
            actual_percentage = 100.0 - relative_percentage
            
            # Find the closest fret or fret range (start from fret 1, not 0 which is the nut)
            for fret_num in range(1, len(FRET_POSITIONS)):
                if fret_num not in FRET_POSITIONS:
                    continue
                
                # Get current and next fret positions
                current_fret_pos = FRET_POSITIONS[fret_num]
                next_fret_pos = FRET_POSITIONS.get(fret_num + 1, 100.0)
                
                # Check if the landmark falls between these frets
                if current_fret_pos <= actual_percentage <= next_fret_pos:
                    covered_frets.add(fret_num)
                    break
    
    return covered_frets


def draw_frets_on_neck(frame, x1, y1, x2, y2, num_frets=20):
    """
    Draw fret lines on the detected guitar neck region.
    
    Args:
        frame: The image frame
        x1, y1, x2, y2: Bounding box coordinates of the neck
        num_frets: Number of frets to draw (default: 20)
    """
    neck_width = x2 - x1
    neck_height = y2 - y1
    
    # Draw fret lines
    for fret_num in range(num_frets + 1):
        if fret_num not in FRET_POSITIONS:
            continue
        
        # Calculate position based on percentage (flipped: 100% - percentage)
        fret_percentage = (100.0 - FRET_POSITIONS[fret_num]) / 100.0
        
        # Assuming the guitar neck is horizontal (width is the length)
        # Flipped so fret 1 is on the right, fret 19 is on the left
        fret_x = int(x1 + (neck_width * fret_percentage))
        
        # Determine line color and thickness
        if fret_num == 0:
            # Nut - thicker, different color (now at the right end)
            color = (255, 255, 255)  # White
            thickness = 3
        elif fret_num == 12:
            # 12th fret (octave) - thicker, highlighted
            color = (0, 255, 255)  # Yellow
            thickness = 2
        else:
            # Regular frets
            color = (200, 200, 200)  # Light gray
            thickness = 1
        
        # Draw vertical line for the fret
        cv2.line(frame, (fret_x, y1), (fret_x, y2), color, thickness)
        
        # Add fret number labels for key frets (0 is Nut, actual frets start at 1)
        if fret_num in [0, 1, 3, 5, 7, 9, 12, 15, 17, 19]:
            label = "Nut" if fret_num == 0 else str(fret_num)
            cv2.putText(
                frame,
                label,
                (fret_x - 10, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1
            )


def main():
    # Use global variable for confidence threshold
    global CONFIDENCE_THRESHOLD
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Complete Guitar Detection: Video + Audio')
    parser.add_argument('video', type=str, nargs='?', default=None,
                       help='Path to video file (if not specified, uses webcam)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save output video with fret annotations')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Skip audio preprocessing (use original audio)')
    args = parser.parse_args()
    
    video_mode = "VIDEO FILE" if args.video else "WEBCAM REAL-TIME"
    
    print("\n" + "=" * 80)
    print(f"{'🎸 COMPLETE GUITAR DETECTION':^80}")
    print(f"{'Mode: ' + video_mode:^80}")
    print("=" * 80)
    
    # Load the trained model
    print("\n┌─ 📦 MODEL INITIALIZATION " + "─" * 51)
    print(f"│ Loading: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("│ Status: ✅ YOLO model loaded successfully")
        print(f"│ Type: YOLOv8")
        print(f"│ Classes: {', '.join(CLASS_NAMES)}")
    except Exception as e:
        print(f"│ Status: ❌ Error - {e}")
        print("└" + "─" * 79)
        return
    print("└" + "─" * 79)
    
    # Initialize cvzone Hand Detector
    print("\n┌─ 👋 HAND DETECTOR INITIALIZATION " + "─" * 43)
    detector = HandDetector(
        detectionCon=0.3,
        minTrackCon=0.3,
        maxHands=1
    )
    print("│ Status: ✅ Hand detector ready")
    print("└" + "─" * 79)
    
    # Initialize video capture (webcam or video file)
    print("\n┌─ 🎥 VIDEO SOURCE " + "─" * 60)
    if args.video:
        print(f"│ File: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("│ Status: ❌ Error - Could not open video file")
            print("└" + "─" * 79)
            return
        print("│ Status: ✅ Video file opened")
        
        # Load audio from video
        print("│")
        print("│ 🎵 Extracting audio...")
        try:
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Extract audio using ffmpeg
            result = subprocess.run(
                ['ffmpeg', '-i', args.video, '-vn', '-acodec', 'pcm_s16le', 
                 '-ar', '22050', '-ac', '1', '-y', temp_audio_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"ffmpeg failed: {result.stderr}")
            
            # Audio preprocessing with Demucs or Spleeter
            processed_audio_path = temp_audio_path
            preprocessing_succeeded = False
            
            if not args.no_preprocess:
                print("│")
                print("│ 🔧 Initializing audio preprocessor...")
                preprocessor = AudioPreprocessor(method='auto')
                
                if preprocessor.is_available():
                    print(f"│ Using: {preprocessor.method}")
                    print("│")
                    
                    def progress_callback(msg):
                        print(f"│ {msg}")
                    
                    print("│ 🎵 Preprocessing audio to remove noise...")
                    print("│ This may take a few minutes...")
                    result_path = preprocessor.preprocess_audio(
                        temp_audio_path,
                        progress_callback=progress_callback
                    )
                    print("│")
                    
                    # Verify preprocessing actually succeeded
                    if result_path and result_path != temp_audio_path and os.path.exists(result_path):
                        processed_audio_path = result_path
                        preprocessing_succeeded = True
                    else:
                        print("│ ⚠️  Preprocessing returned original path - may have failed")
                else:
                    print("│ ⚠️  No preprocessor available (Demucs/Spleeter not installed)")
                    print("│ Using original audio")
                    print("│")
            else:
                print("│ ⚠️  Audio preprocessing skipped (--no-preprocess flag)")
                print("│")
            
            # Load the processed audio
            audio_y, audio_sr = librosa.load(processed_audio_path, sr=None)
            has_audio = True
            
            # Show which audio is being used (accurate status)
            if preprocessing_succeeded:
                print("│ 📢 Using: Cleaned audio from preprocessing")
                print(f"│ 📁 Path: {processed_audio_path}")
            else:
                reason = "(--no-preprocess flag)" if args.no_preprocess else "(preprocessing unavailable/failed)"
                print(f"│ 📢 Using: Original audio {reason}")
            
            # Clean up temp files
            if processed_audio_path != temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            
            print(f"│ Duration: {len(audio_y)/audio_sr:.2f}s at {audio_sr} Hz")
            print("│ Status: ✅ Audio ready")
        except Exception as e:
            print(f"│ Status: ⚠️  Audio unavailable - {e}")
            has_audio = False
    else:
        print("│ Source: Webcam (device 0)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("│ Status: ❌ Error - Could not open webcam")
            print("└" + "─" * 79)
            return
        # Set camera resolution (lower for better FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("│ Status: ✅ Webcam ready")
        has_audio = False
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.video else 30
    
    print(f"│ Resolution: {actual_width}x{actual_height}")
    if args.video:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"│ Frames: {total_frames}")
        print(f"│ FPS: {fps}")
    
    # Setup video writer if output path is provided
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (actual_width, actual_height))
        print(f"│ Output: {args.output}")
    print("└" + "─" * 79)
    
    print("\n┌─ ⌨️  CONTROLS " + "─" * 63)
    print("│ q          Quit")
    if not args.video:
        print("│ s          Save current frame")
    print("│ +/-        Adjust confidence threshold")
    print("│ h          Toggle hand landmarks")
    if args.video:
        print("│ SPACE      Pause/Resume")
    print("└" + "─" * 79)
    
    print("\n" + "=" * 80)
    print(f"{'🎬 PROCESSING':^80}")
    print("=" * 80 + "\n")
    
    # Hand detection toggle
    show_hands = True
    
    # Video playback control
    paused = False
    
    # FPS calculation
    fps_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    # Detection counter
    frame_count = 0
    saved_count = 0
    
    # Performance optimization
    hand_detection_skip = 0  # Process hands every N frames
    hands_list = []  # Cache hand detection results
    
    # Track all frets covered throughout the video
    all_frets_covered = set()
    
    # Audio frequency tracking
    all_audio_frets = {}  # time -> (hz, string, fret)
    
    # Temporal smoothing for nut detection
    last_nut_position = None
    nut_detection_history = []
    max_nut_history = 5  # Keep last 5 detections for smoothing
    
    try:
        while True:
            # Handle pause in video mode
            if paused and args.video:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    paused = False
                    print("▶️  Resumed")
                elif key == ord('q'):
                    break
                continue
            
            # Read frame from video/webcam
            ret, frame = cap.read()
            
            if not ret:
                if args.video:
                    print("\n✅ Video processing complete")
                else:
                    print("❌ Error: Could not read frame from webcam")
                break
            
            frame_count += 1
            
            # Analyze audio if available
            current_hz = None
            audio_string = None
            audio_fret = None
            if has_audio and args.video:
                current_time = frame_count / fps
                start_sample = int(current_time * audio_sr)
                end_sample = int((current_time + 1/fps) * audio_sr)
                
                if end_sample < len(audio_y):
                    audio_segment = audio_y[start_sample:end_sample]
                    detected_hz, string, fret, exact_hz = analyze_audio_segment(audio_segment, audio_sr)
                    
                    if detected_hz is not None:
                        current_hz = detected_hz
                        audio_string = string
                        audio_fret = fret
                        all_audio_frets[current_time] = (detected_hz, string, fret)
                        print(f"[{current_time:6.2f}s] 🎵 Audio: {detected_hz:6.2f}Hz → S{string}F{fret:2d}", end="")
            
            # Resize frame for faster processing
            process_frame = cv2.resize(frame, (640, 480))
            
            # Process hand detection with cvzone (every 4 frames for performance)
            hand_detection_skip += 1
            if hand_detection_skip >= 4:
                hands_list, _ = detector.findHands(process_frame, draw=False)
                hand_detection_skip = 0
            
            # Run YOLO detection with optimized image size - use lower conf for nut
            results = model.predict(
                process_frame,
                conf=NUT_CONFIDENCE_THRESHOLD,  # Lower threshold to catch nut
                iou=IOU_THRESHOLD,
                imgsz=416,  # Smaller image size for faster inference
                verbose=False
            )
            
            # Process results
            detections = results[0].boxes
            detection_count = len(detections)
            hands_detected = 0
            
            # Scale factor for coordinate conversion
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            
            # Store neck coordinates for fret detection
            neck_coords = None
            
            # Track nut detection for this frame
            nut_detected_this_frame = False
            current_nut_position = None
            
            # First pass: detect neck and nut
            for box in detections:
                # Get box coordinates and scale to original frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Get class and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
                
                # Get color for this class
                color = COLORS.get(class_name, (0, 255, 0))
                
                # Process neck detection (with higher confidence threshold)
                if class_name == 'neck' and conf >= CONFIDENCE_THRESHOLD:
                    draw_frets_on_neck(frame, x1, y1, x2, y2)
                    # Store neck coordinates for fret detection
                    neck_coords = (x1, y1, x2, y2)
                    
                    # Draw bounding box for neck
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Process nut detection (with lower threshold and temporal smoothing)
                elif class_name == 'nut' and conf >= NUT_CONFIDENCE_THRESHOLD:
                    nut_detected_this_frame = True
                    current_nut_position = (x1, y1, x2, y2, conf)
            
            # Temporal smoothing for nut: use last known position if no detection this frame
            if nut_detected_this_frame:
                nut_detection_history.append(current_nut_position)
                if len(nut_detection_history) > max_nut_history:
                    nut_detection_history.pop(0)
                last_nut_position = current_nut_position
            elif last_nut_position is not None and len(nut_detection_history) > 0:
                current_nut_position = last_nut_position
            
            # Draw nut if we have a position (detected or from history)
            if current_nut_position is not None:
                nut_x1, nut_y1, nut_x2, nut_y2, nut_conf = current_nut_position
                nut_color = COLORS.get('nut', (255, 0, 255))
                cv2.rectangle(frame, (nut_x1, nut_y1), (nut_x2, nut_y2), nut_color, 3)
                status = "LIVE" if nut_detected_this_frame else "TRACK"
                cv2.putText(frame, f"NUT [{status}]: {nut_conf:.2f}", (nut_x1, nut_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw hand landmarks if detected
            covered_frets_all = set()
            if show_hands and hands_list:
                hands_detected = len(hands_list)
                
                for hand in hands_list:
                    # Get hand type (Left/Right)
                    hand_type = hand["type"]  # "Left" or "Right"
                    
                    # Get landmark list (21 landmarks, each with x, y, z)
                    lmList = hand["lmList"]  # List of 21 landmarks
                    
                    # Detect which frets are covered by this hand
                    if neck_coords:
                        covered_frets = get_frets_from_hand_position(
                            lmList, neck_coords[0], neck_coords[1], 
                            neck_coords[2], neck_coords[3], scale_x, scale_y
                        )
                        covered_frets_all.update(covered_frets)
                        all_frets_covered.update(covered_frets)
                        
                        # Print covered frets if any
                        if covered_frets:
                            fret_list = sorted(list(covered_frets))
                            print(f" | 👋 {hand_type} hand: Frets {fret_list}", end="")
                            
                            # If audio is detected at this moment, match it with covered frets
                            if current_hz is not None and audio_string is not None and audio_fret is not None:
                                # Check if the audio-detected fret is in the covered frets
                                if audio_fret in covered_frets:
                                    print(f" | ✅ MATCH")
                                else:
                                    # Find which strings could be played on the covered frets
                                    possible_matches = []
                                    for fret in covered_frets:
                                        for string in range(1, 7):
                                            if (string, fret) in GUITAR_FREQUENCIES:
                                                freq = GUITAR_FREQUENCIES[(string, fret)]
                                                if abs(freq - current_hz) <= 5.0:  # Within 5Hz tolerance
                                                    possible_matches.append((string, fret, freq))
                                    
                                    if possible_matches:
                                        s, f, freq = possible_matches[0]
                                        print(f" | ✅ MATCH (S{s}F{f})")
                                    else:
                                        print(f" | ⚠️  No match")
                            else:
                                print()
                    
                    # Get wrist position (landmark 0) for label and scale to original frame
                    wrist_x, wrist_y = int(lmList[0][0] * scale_x), int(lmList[0][1] * scale_y)
                    
                    # Draw hand label
                    cv2.putText(
                        frame,
                        f"{hand_type} Hand",
                        (wrist_x - 50, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    
                    # Draw hand skeleton connections
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                        (5, 9), (9, 13), (13, 17)  # Palm
                    ]
                    
                    for connection in connections:
                        p1_x = int(lmList[connection[0]][0] * scale_x)
                        p1_y = int(lmList[connection[0]][1] * scale_y)
                        p2_x = int(lmList[connection[1]][0] * scale_x)
                        p2_y = int(lmList[connection[1]][1] * scale_y)
                        cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 2)
                    
                    # Highlight fingertips (landmarks: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky)
                    fingertip_indices = [4, 8, 12, 16, 20]
                    
                    for tip_idx in fingertip_indices:
                        tip_x, tip_y = int(lmList[tip_idx][0] * scale_x), int(lmList[tip_idx][1] * scale_y)
                        cv2.circle(frame, (tip_x, tip_y), 8, (255, 0, 255), -1)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Draw info panel on frame
            info_y = 30
            cv2.putText(frame, f"FPS: {fps_display}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 30
            cv2.putText(frame, f"Neck Detections: {detection_count}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 30
            cv2.putText(frame, f"Hands Detected: {hands_detected}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_y += 30
            cv2.putText(frame, f"Confidence: {CONFIDENCE_THRESHOLD:.2f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display covered frets from hand
            if covered_frets_all:
                info_y += 30
                fret_text = f"Hand Frets: {sorted(list(covered_frets_all))}"
                cv2.putText(frame, fret_text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display audio frequency and fret
            if current_hz is not None:
                info_y += 30
                audio_text = f"Audio: {current_hz:.1f}Hz - S{audio_string}F{audio_fret}"
                
                # Check if audio matches any covered fret
                is_match = False
                if covered_frets_all:
                    # Check if audio fret is in covered frets
                    if audio_fret in covered_frets_all:
                        is_match = True
                        audio_text += " ✓MATCH"
                    else:
                        # Check if audio frequency matches any string on covered frets
                        for fret in covered_frets_all:
                            for string in range(1, 7):
                                if (string, fret) in GUITAR_FREQUENCIES:
                                    freq = GUITAR_FREQUENCIES[(string, fret)]
                                    if abs(freq - current_hz) <= 5.0:
                                        is_match = True
                                        audio_text = f"Audio: {current_hz:.1f}Hz - S{string}F{fret} ✓MATCH"
                                        break
                            if is_match:
                                break
                
                color = (0, 255, 0) if is_match else (255, 0, 255)
                cv2.putText(frame, audio_text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show video progress for video files
            if args.video:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                info_y += 30
                cv2.putText(frame, f"Progress: {progress:.1f}%", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            info_y += 30
            cv2.putText(frame, "Press 'q' to quit", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to output video if specified
            if video_writer is not None:
                video_writer.write(frame)
            
            # Display the frame
            window_title = 'Complete Guitar Detection - Video' if args.video else 'Complete Guitar Detection - Webcam'
            cv2.imshow(window_title, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n" + "─" * 80)
                print("Quitting...")
                break
            elif key == ord('s') and not args.video:
                # Save current frame (webcam only)
                saved_count += 1
                filename = f"detection_capture_{saved_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
            elif key == ord(' ') and args.video:
                # Pause/resume (video only)
                paused = not paused
                status = "⏸️  PAUSED" if paused else "▶️  PLAYING"
                print(f"\r{status}", end="\n" if paused else "")
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                CONFIDENCE_THRESHOLD = max(0.05, CONFIDENCE_THRESHOLD - 0.05)
                print(f"⚙️  Confidence: {CONFIDENCE_THRESHOLD:.2f}")
            elif key == ord('h'):
                # Toggle hand landmarks
                show_hands = not show_hands
                print(f"👋 Hands: {'ON' if show_hands else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 80)
        print(f"{'📊 SESSION SUMMARY':^80}")
        print("=" * 80)
        
        print("\n┌─ PERFORMANCE " + "─" * 64)
        print(f"│ Frames processed: {frame_count}")
        if not args.video:
            print(f"│ Frames saved: {saved_count}")
        print(f"│ Average FPS: ~{fps_display}")
        if video_writer is not None:
            print(f"│ Output saved: {args.output}")
        print("└" + "─" * 79)
        
        # Visual fret summary
        print("\n┌─ 👋 HAND DETECTION " + "─" * 57)
        if all_frets_covered:
            fret_list = sorted(list(all_frets_covered))
            print(f"│ Frets covered: {fret_list}")
            print(f"│ Total unique frets: {len(fret_list)}")
        else:
            print("│ No frets detected")
        print("└" + "─" * 79)
        
        # Audio fret summary
        print("\n┌─ 🎵 AUDIO ANALYSIS " + "─" * 57)
        if all_audio_frets:
            unique_audio_frets = set()
            for hz, string, fret in all_audio_frets.values():
                unique_audio_frets.add((string, fret))
            
            print(f"│ Detections: {len(all_audio_frets)} timestamps")
            print(f"│ Unique frets: {len(unique_audio_frets)}")
            print("│")
            print("│ Detected notes:")
            for string, fret in sorted(unique_audio_frets):
                freq = GUITAR_FREQUENCIES.get((string, fret), 0)
                print(f"│   String {string}, Fret {fret:2d}  →  {freq:7.2f} Hz")
        else:
            print("│ No audio frequencies detected")
        print("└" + "─" * 79)
        
        print("\n" + "=" * 80)
        print(f"{'✅ COMPLETE':^80}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
