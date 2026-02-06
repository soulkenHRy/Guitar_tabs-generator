#!/usr/bin/env python3
"""
Guitar Detector GUI
Simple interface to upload video files and run detection with built-in tab generation
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
import subprocess
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from scipy.io import wavfile
from ultralytics import YOLO
from cvzone.HandTrackingModule import HandDetector
import time
from guitar_tuner import analyze_audio_segment, GUITAR_FREQUENCIES, find_closest_note
from audio_preprocessor import AudioPreprocessor
import tempfile
from PIL import Image, ImageTk


# Guitar Detection Configuration
MODEL_PATH = "guitar_detector_best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class names for the new model: 0=fret, 1=neck, 2=nut
CLASS_NAMES = ['fret', 'neck', 'nut']

# Colors for each class (BGR format)
COLORS = {
    'fret': (0, 255, 255),            # Yellow for frets
    'neck': (0, 255, 0),              # Green for neck
    'nut': (255, 255, 255),           # White for nut
}

# Fret positions as percentage of fretboard length
FRET_POSITIONS = {
    0: 0.0, 1: 8.17, 2: 15.88, 3: 23.18, 4: 30.10, 5: 36.66,
    6: 42.89, 7: 48.83, 8: 54.48, 9: 59.86, 10: 65.00, 11: 69.92,
    12: 72.75, 13: 76.84, 14: 80.70, 15: 84.35, 16: 87.80,
    17: 91.08, 18: 94.20, 19: 97.16, 20: 100.0,
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

# Audio analysis functions are now imported from guitar_tuner module
# which uses accurate FFT + autocorrelation pitch detection


class FretStabilizer:
    """Stabilizes fret positions across frames using exponential moving average."""
    
    def __init__(self, smoothing_factor=0.05, max_frets=24):
        """
        Args:
            smoothing_factor: How much to blend new detections (0.0-1.0). 
                              Lower = more stable, higher = more responsive
                              0.05 = very stable, almost locked in place
            max_frets: Maximum number of frets to track
        """
        self.smoothing_factor = smoothing_factor
        self.max_frets = max_frets
        self.stable_frets = {}  # fret_num -> (x1, y1, x2, y2) - starts EMPTY
        self.stable_nut = None
        self.stable_neck = None
        self.frames_since_detection = 0
        self.max_frames_without_detection = 15  # Only keep for 0.5 seconds if no detection
        self.initialized = False  # Only show frets after first detection
        
    def update(self, detected_frets, nut_coords=None, neck_coords=None):
        """Update stable positions with new detections.
        
        Only updates positions when the whole guitar moves (all frets shift).
        If any fret remains in the same position, it means the guitar is stationary
        and we should keep all frets locked.
        
        Args:
            detected_frets: List of (fret_number, x1, y1, x2, y2) tuples
            nut_coords: Optional (x1, y1, x2, y2) for nut
            neck_coords: Optional (x1, y1, x2, y2) for neck
            
        Returns:
            Tuple of (stable_frets_list, stable_nut, stable_neck)
        """
        if detected_frets:
            self.frames_since_detection = 0
            
            # Build a map of current detections
            current_fret_map = {f[0]: (f[1], f[2], f[3], f[4]) for f in detected_frets}
            
            # Check if guitar has moved by comparing detected frets to stable frets
            # Calculate average shift of all matching frets
            shifts = []
            stationary_count = 0
            movement_threshold = 5  # pixels - frets within this are considered "same position"
            
            for fret_num, coords in current_fret_map.items():
                if fret_num in self.stable_frets:
                    old_coords = self.stable_frets[fret_num]
                    # Calculate center x shift
                    old_center_x = (old_coords[0] + old_coords[2]) // 2
                    new_center_x = (coords[0] + coords[2]) // 2
                    shift = abs(new_center_x - old_center_x)
                    shifts.append(shift)
                    
                    if shift < movement_threshold:
                        stationary_count += 1
            
            # If we have stable frets to compare with
            if self.stable_frets:
                # If ANY fret is stationary, the guitar hasn't moved
                # Keep all existing stable positions, don't update
                if stationary_count > 0:
                    # Guitar is stationary - keep all existing frets locked
                    # Only add NEW frets that weren't tracked before
                    for fret_num, coords in current_fret_map.items():
                        if fret_num not in self.stable_frets:
                            self.stable_frets[fret_num] = coords
                else:
                    # ALL frets have shifted - guitar has moved
                    # Update all positions with smoothing
                    for fret_num, coords in current_fret_map.items():
                        x1, y1, x2, y2 = coords
                        if fret_num in self.stable_frets:
                            old = self.stable_frets[fret_num]
                            alpha = self.smoothing_factor * 3  # Faster update when guitar moves
                            alpha = min(alpha, 0.5)  # Cap at 50%
                            new_coords = (
                                int(old[0] * (1 - alpha) + x1 * alpha),
                                int(old[1] * (1 - alpha) + y1 * alpha),
                                int(old[2] * (1 - alpha) + x2 * alpha),
                                int(old[3] * (1 - alpha) + y2 * alpha)
                            )
                            self.stable_frets[fret_num] = new_coords
                        else:
                            self.stable_frets[fret_num] = coords
                    
                    # Remove frets no longer detected when guitar moves
                    old_frets = list(self.stable_frets.keys())
                    max_detected = max(current_fret_map.keys()) if current_fret_map else 0
                    for old_fret in old_frets:
                        if old_fret not in current_fret_map and old_fret > max_detected:
                            del self.stable_frets[old_fret]
            else:
                # First time - add all detected frets
                for fret_num, coords in current_fret_map.items():
                    self.stable_frets[fret_num] = coords
        else:
            self.frames_since_detection += 1
        
        # Update nut position
        if nut_coords:
            if self.stable_nut:
                alpha = self.smoothing_factor
                self.stable_nut = (
                    int(self.stable_nut[0] * (1 - alpha) + nut_coords[0] * alpha),
                    int(self.stable_nut[1] * (1 - alpha) + nut_coords[1] * alpha),
                    int(self.stable_nut[2] * (1 - alpha) + nut_coords[2] * alpha),
                    int(self.stable_nut[3] * (1 - alpha) + nut_coords[3] * alpha)
                )
            else:
                self.stable_nut = nut_coords
        
        # Update neck position
        if neck_coords:
            if self.stable_neck:
                alpha = self.smoothing_factor
                self.stable_neck = (
                    int(self.stable_neck[0] * (1 - alpha) + neck_coords[0] * alpha),
                    int(self.stable_neck[1] * (1 - alpha) + neck_coords[1] * alpha),
                    int(self.stable_neck[2] * (1 - alpha) + neck_coords[2] * alpha),
                    int(self.stable_neck[3] * (1 - alpha) + neck_coords[3] * alpha)
                )
            else:
                self.stable_neck = neck_coords
        
        # Clear if no detection for too long
        if self.frames_since_detection > self.max_frames_without_detection:
            self.stable_frets.clear()
            self.stable_nut = None
            self.stable_neck = None
            self.initialized = False
        
        # Only return frets if we have actual detections (not pre-positioned)
        if not self.initialized and not detected_frets:
            return [], None, None
        
        if detected_frets:
            self.initialized = True
        
        # Return stable positions as list
        stable_frets_list = [(fret_num, *coords) for fret_num, coords in sorted(self.stable_frets.items())]
        return stable_frets_list, self.stable_nut, self.stable_neck
    
    def reset(self):
        """Reset all tracked positions."""
        self.stable_frets.clear()
        self.stable_nut = None
        self.stable_neck = None
        self.frames_since_detection = 0
        self.initialized = False


class FretGapTracker:
    """
    Tracks gaps between fret bars to correctly assign fret numbers even when frets are occluded.
    
    This works BEFORE fret number assignment by:
    1. Learning the expected gap pattern between frets (they get smaller up the neck)
    2. Detecting when a gap is abnormally large (indicates hidden fret)
    3. Assigning correct fret numbers that account for missing frets
    4. Optionally inserting hidden frets at estimated positions (disabled by default)
    
    IMPORTANT: No pre-positioning - all tracking only starts after frets are detected.
    """
    
    def __init__(self, max_frets=24, stability_frames=12, occlusion_ratio=1.85, restore_hidden=False):
        """
        Args:
            max_frets: Maximum number of frets to track
            stability_frames: Frames needed before gaps are considered "stable" (12 = ~0.4sec at 30fps)
            occlusion_ratio: Gap increase ratio that indicates occlusion (1.85 = must be ~2x normal gap)
            restore_hidden: Whether to restore hidden frets visually (disabled to prevent false positives)
        """
        self.max_frets = max_frets
        self.stability_frames = stability_frames
        self.occlusion_ratio = occlusion_ratio
        self.restore_hidden = restore_hidden
        
        # Track stable gap sizes in order
        self.stable_gaps = []  # List of stable gap sizes in order
        self.gap_confidence = 0  # How many frames we've seen consistent gaps
        self.last_good_frets = {}  # fret_num -> (x1, y1, x2, y2)
        self.last_fret_count = 0  # Expected number of frets
        
        self.initialized = False
        self.min_frets_for_tracking = 4  # Need at least 4 frets to start tracking
        
    def _get_fret_centers(self, fret_boxes):
        """Get sorted center x positions of fret boxes."""
        centers = []
        for box in fret_boxes:
            center_x = (box[0] + box[2]) // 2
            centers.append(center_x)
        return sorted(centers)
    
    def _calculate_gaps(self, centers):
        """Calculate gaps between consecutive fret centers."""
        gaps = []
        for i in range(len(centers) - 1):
            gaps.append(abs(centers[i + 1] - centers[i]))
        return gaps
    
    def _find_missing_frets(self, current_gaps):
        """
        Detect missing frets by comparing current gaps to stable gaps.
        
        Returns: List of indices where frets appear to be missing
        """
        if not self.stable_gaps or len(current_gaps) == 0:
            return []
        
        missing_at = []
        
        # Compare each current gap to what we expect
        # If current gap is much larger than expected, a fret is missing
        stable_idx = 0
        for curr_idx, curr_gap in enumerate(current_gaps):
            if stable_idx >= len(self.stable_gaps):
                break
                
            expected_gap = self.stable_gaps[stable_idx]
            
            # Check if this gap is too large (fret missing)
            if curr_gap > expected_gap * self.occlusion_ratio:
                # This gap spans where a fret should be
                # Check if it's roughly 2x the expected gap (one fret missing)
                if curr_gap > expected_gap * 1.8 and stable_idx + 1 < len(self.stable_gaps):
                    combined_expected = expected_gap + self.stable_gaps[stable_idx + 1]
                    if curr_gap < combined_expected * 1.3:
                        # Yes, one fret is missing here
                        missing_at.append(curr_idx)
                        stable_idx += 2  # Skip the missing fret's gap
                        continue
            
            stable_idx += 1
        
        return missing_at
    
    def assign_fret_numbers(self, fret_boxes, nut_x=None):
        """
        Assign fret numbers to detected boxes, accounting for hidden frets.
        
        Args:
            fret_boxes: List of (x1, y1, x2, y2, conf) raw detections
            nut_x: X position of nut center (for sorting direction)
            
        Returns:
            detected_frets: List of (fret_number, x1, y1, x2, y2)
            hidden_frets: List of (fret_number, x1, y1, x2, y2) for frets that were hidden but restored
        """
        if not fret_boxes:
            return [], []
        
        # Sort fret boxes by position relative to nut
        if nut_x is not None:
            # Sort by distance from nut
            sorted_boxes = sorted(fret_boxes, key=lambda f: abs((f[0] + f[2]) // 2 - nut_x))
            # Determine direction (nut left or right)
            first_fret_x = (sorted_boxes[0][0] + sorted_boxes[0][2]) // 2
            nut_on_left = nut_x < first_fret_x
        else:
            # Assume nut on left, sort by x position
            sorted_boxes = sorted(fret_boxes, key=lambda f: (f[0] + f[2]) // 2)
            nut_on_left = True
        
        # Get centers and gaps
        centers = [(box[0] + box[2]) // 2 for box in sorted_boxes]
        current_gaps = self._calculate_gaps(centers)
        
        detected_frets = []
        hidden_frets = []
        
        # Not enough frets to track gaps - just number sequentially
        if len(fret_boxes) < self.min_frets_for_tracking:
            for fret_num, (fx1, fy1, fx2, fy2, fconf) in enumerate(sorted_boxes, start=1):
                detected_frets.append((fret_num, fx1, fy1, fx2, fy2))
            return detected_frets, hidden_frets
        
        # Initialize stable gaps if not yet done
        if not self.initialized:
            self.stable_gaps = current_gaps.copy()
            self.gap_confidence = 1
            self.last_fret_count = len(fret_boxes)
            self.initialized = True
            
            # Just number sequentially for first frame
            for fret_num, (fx1, fy1, fx2, fy2, fconf) in enumerate(sorted_boxes, start=1):
                detected_frets.append((fret_num, fx1, fy1, fx2, fy2))
                self.last_good_frets[fret_num] = (fx1, fy1, fx2, fy2)
            return detected_frets, hidden_frets
        
        # Check if we have fewer frets than before (possible occlusion)
        fret_count_diff = self.last_fret_count - len(fret_boxes)
        
        # Find where frets might be missing - BUT only if we have enough confidence
        missing_indices = []
        
        # Only detect occlusion if:
        # 1. We have built enough confidence (stable gaps for several frames)
        # 2. The fret count actually dropped
        # 3. We have stable gaps to compare against
        if (fret_count_diff > 0 and 
            len(self.stable_gaps) > 0 and 
            self.gap_confidence >= self.stability_frames):
            
            # Compare gaps to find abnormally large ones
            stable_idx = 0
            for curr_idx, curr_gap in enumerate(current_gaps):
                if stable_idx >= len(self.stable_gaps):
                    break
                
                expected = self.stable_gaps[stable_idx]
                
                # Gap is much larger than expected (must be nearly double = one fret hidden)
                if curr_gap > expected * self.occlusion_ratio:
                    # Estimate how many frets are missing
                    accumulated = 0
                    frets_missing = 0
                    while stable_idx + frets_missing < len(self.stable_gaps):
                        accumulated += self.stable_gaps[stable_idx + frets_missing]
                        frets_missing += 1
                        # If accumulated expected gaps match current gap, we found the missing count
                        if accumulated > curr_gap * 0.7:
                            break
                    
                    if frets_missing > 1:
                        # Record that frets are missing after this position
                        missing_indices.append((curr_idx, frets_missing - 1))
                        stable_idx += frets_missing
                        continue
                
                stable_idx += 1
        
        # Assign fret numbers with gaps for missing frets
        current_fret_num = 1
        for i, (fx1, fy1, fx2, fy2, fconf) in enumerate(sorted_boxes):
            # Check if frets are missing before this position
            for miss_idx, miss_count in missing_indices:
                if i == miss_idx + 1:  # Missing frets are between miss_idx and miss_idx+1
                    # Skip fret numbers for missing frets (correct numbering)
                    for j in range(miss_count):
                        hidden_fret_num = current_fret_num
                        # Only restore visual position if enabled (disabled by default to prevent false positives)
                        if self.restore_hidden and hidden_fret_num in self.last_good_frets:
                            coords = self.last_good_frets[hidden_fret_num]
                            hidden_frets.append((hidden_fret_num, *coords))
                            print(f"[FretGapTracker] Fret {hidden_fret_num} hidden - restored from memory")
                        else:
                            # Just skip the number, don't insert visual fret
                            print(f"[FretGapTracker] Fret {hidden_fret_num} appears hidden (skipping number)")
                        current_fret_num += 1
            
            detected_frets.append((current_fret_num, fx1, fy1, fx2, fy2))
            self.last_good_frets[current_fret_num] = (fx1, fy1, fx2, fy2)
            current_fret_num += 1
        
        # Update stable gaps if detection looks consistent (no missing frets)
        if len(missing_indices) == 0 and len(current_gaps) == len(self.stable_gaps):
            # Gaps are consistent, build confidence
            self.gap_confidence = min(self.gap_confidence + 1, self.stability_frames * 2)
            
            # Slowly update stable gaps with exponential moving average
            if self.gap_confidence >= self.stability_frames:
                alpha = 0.1
                for i in range(len(self.stable_gaps)):
                    self.stable_gaps[i] = int(self.stable_gaps[i] * (1 - alpha) + current_gaps[i] * alpha)
        elif len(missing_indices) == 0 and len(fret_boxes) > self.last_fret_count:
            # More frets visible now - update our baseline
            self.stable_gaps = current_gaps.copy()
            self.last_fret_count = len(fret_boxes)
            self.gap_confidence = 1
        
        # Update last fret count (only if no occlusion detected)
        if len(missing_indices) == 0:
            self.last_fret_count = max(self.last_fret_count, len(fret_boxes))
        
        return detected_frets, hidden_frets
    
    def reset(self):
        """Reset all tracking state."""
        self.stable_gaps.clear()
        self.gap_confidence = 0
        self.last_good_frets.clear()
        self.last_fret_count = 0
        self.initialized = False


def get_frets_from_hand_position(hand_lmList, detected_frets, neck_coords, nut_coords, scale_x, scale_y):
    """Determine which fret a hand is pressing.
    
    Fret numbering:
    - Fret 1 = space from NUT to 1st fret bar
    - Fret 2 = space from 1st bar to 2nd bar
    - Fret 3 = space from 2nd bar to 3rd bar
    - etc.
    
    Args:
        hand_lmList: List of hand landmarks
        detected_frets: List of (fret_number, x1, y1, x2, y2) tuples for detected fret bars
        neck_coords: (x1, y1, x2, y2) of the neck bounding box
        nut_coords: (x1, y1, x2, y2) of the nut, or None
        scale_x, scale_y: Scaling factors from process frame to original frame
    
    Returns:
        Set of fret numbers that the hand is pressing
    """
    covered_frets = set()
    
    if not neck_coords:
        return covered_frets
    
    neck_x1, neck_y1, neck_x2, neck_y2 = neck_coords
    
    # Get nut position (if detected) as the starting reference
    if nut_coords:
        nut_x = (nut_coords[0] + nut_coords[2]) // 2
    else:
        # If no nut, assume it's at the left edge of neck
        nut_x = neck_x1
    
    # Get fret bar x-positions (center of each bar), sorted by distance from nut
    bar_positions = []
    for fret_num, fx1, fy1, fx2, fy2 in detected_frets:
        center_x = (fx1 + fx2) // 2
        bar_positions.append(center_x)
    
    # Sort bars by distance from nut
    bar_positions.sort(key=lambda x: abs(x - nut_x))
    
    for landmark in hand_lmList:
        lm_x = int(landmark[0] * scale_x)
        lm_y = int(landmark[1] * scale_y)
        
        # Check if landmark is within neck bounds
        if not (neck_x1 <= lm_x <= neck_x2 and neck_y1 <= lm_y <= neck_y2):
            continue
        
        # Determine which fret space the landmark is in
        # Fret 1 = between nut and first bar
        # Fret 2 = between first bar and second bar, etc.
        
        # Check if before first bar (fret 1)
        if not bar_positions:
            continue
            
        # Determine direction: is nut to the left or right of bars?
        first_bar = bar_positions[0]
        if nut_x < first_bar:
            # Nut is on the left, bars go right
            if lm_x < first_bar:
                # Between nut and first bar = fret 1
                covered_frets.add(1)
            else:
                # Find which space between bars
                for i in range(len(bar_positions)):
                    if i == len(bar_positions) - 1:
                        # Past last bar
                        covered_frets.add(i + 2)
                    elif lm_x < bar_positions[i + 1]:
                        # Between bar[i] and bar[i+1] = fret (i+2)
                        covered_frets.add(i + 2)
                        break
        else:
            # Nut is on the right, bars go left
            if lm_x > first_bar:
                # Between nut and first bar = fret 1
                covered_frets.add(1)
            else:
                # Find which space between bars
                for i in range(len(bar_positions)):
                    if i == len(bar_positions) - 1:
                        # Past last bar
                        covered_frets.add(i + 2)
                    elif lm_x > bar_positions[i + 1]:
                        # Between bar[i] and bar[i+1] = fret (i+2)
                        covered_frets.add(i + 2)
                        break
    
    return covered_frets


def draw_detected_frets(frame, detected_frets, nut_coords=None, neck_coords=None):
    """Draw detected fret bars as bold lines on the frame.
    
    Args:
        frame: The frame to draw on
        detected_frets: List of (fret_number, x1, y1, x2, y2) tuples for fret bars
        nut_coords: Optional (x1, y1, x2, y2) tuple for nut position
        neck_coords: Optional (x1, y1, x2, y2) tuple for neck position
    """
    # Draw nut if detected - BOLD white bar
    if nut_coords:
        x1, y1, x2, y2 = nut_coords
        center_x = (x1 + x2) // 2
        cv2.line(frame, (center_x, y1 - 5), (center_x, y2 + 5), (255, 255, 255), 4)
        cv2.putText(frame, "NUT", (center_x - 20, y1 - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw each detected fret bar as a BOLD vertical line
    for fret_num, x1, y1, x2, y2 in detected_frets:
        center_x = (x1 + x2) // 2
        
        # Color based on fret importance - all are bold and visible
        if fret_num == 12:
            color = (0, 255, 255)  # Bright yellow for 12th fret
            thickness = 4
        elif fret_num in [3, 5, 7, 9, 15, 17, 19]:
            color = (255, 200, 0)  # Cyan for marker frets
            thickness = 3
        else:
            color = (200, 200, 200)  # Bright gray for regular frets
            thickness = 2
        
        # Draw fret as a bold vertical line
        cv2.line(frame, (center_x, y1), (center_x, y2), color, thickness)
        
        # Draw fret number above - larger and bolder
        cv2.putText(frame, str(fret_num), (center_x - 8, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class GuitarTabGenerator:
    """Generates guitar tablature from detected string/fret positions."""
    
    def __init__(self, callback=None):
        self.notes = []
        self.tab_lines = [[] for _ in range(6)]
        self.string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        self.callback = callback
        self.last_match = None  # Track last match to filter consecutive duplicates
        
    def add_note(self, timestamp, string_num, fret_num):
        """Add a detected note to the tab, filtering consecutive duplicates."""
        # Skip if this is the exact same match as the last one
        current_match = (string_num, fret_num)
        if self.last_match == current_match:
            return
        
        # Update last match and add the note
        self.last_match = current_match
        self.notes.append((timestamp, string_num, fret_num))
        if self.callback:
            self.callback(timestamp, string_num, fret_num)
        
    def generate_tab(self, notes_per_line=20):
        """Generate ASCII guitar tablature from collected notes."""
        if not self.notes:
            return "No notes detected"
        
        sorted_notes = sorted(self.notes, key=lambda x: x[0])
        tab_output = []
        tab_output.append("="*60)
        tab_output.append("GUITAR TABLATURE")
        tab_output.append("="*60)
        tab_output.append(f"Total notes: {len(sorted_notes)}")
        tab_output.append("")
        
        lines = [[] for _ in range(6)]
        for timestamp, string_num, fret_num in sorted_notes:
            line_idx = string_num - 1
            for i in range(6):
                if i == line_idx:
                    lines[i].append(str(fret_num))
                else:
                    lines[i].append('-')
        
        for start_pos in range(0, len(lines[0]), notes_per_line):
            end_pos = min(start_pos + notes_per_line, len(lines[0]))
            
            for string_idx, string_name in enumerate(self.string_names):
                tab_line = f"{string_name}|--"
                for note_pos in range(start_pos, end_pos):
                    fret_val = lines[string_idx][note_pos]
                    if fret_val == '-':
                        tab_line += "--"
                    else:
                        if len(fret_val) == 1:
                            tab_line += fret_val + "-"
                        else:
                            tab_line += fret_val
                tab_line += "|"
                tab_output.append(tab_line)
            tab_output.append("")
        
        return "\n".join(tab_output)
    
    def generate_detailed_list(self):
        """Generate a detailed list of all detected notes with timing."""
        if not self.notes:
            return "No notes detected"
        
        output = []
        output.append("="*60)
        output.append("DETECTED NOTES (Chronological)")
        output.append("="*60)
        output.append(f"{'Time':<12} {'String':<10} {'Fret':<8} {'Frequency'}")
        output.append("-"*60)
        
        for timestamp, string_num, fret_num in sorted(self.notes, key=lambda x: x[0]):
            freq = GUITAR_FREQUENCIES.get((string_num, fret_num), 0)
            time_str = f"{timestamp:.2f}s"
            output.append(f"{time_str:<12} String {string_num:<4} Fret {fret_num:<4} {freq:>7.2f} Hz")
        
        return "\n".join(output)
    
    def save_to_file(self, filename=None):
        """Save the generated tab to a text file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"guitar_tab_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(self.generate_detailed_list())
            f.write("\n\n")
            f.write(self.generate_tab())
        
        return filename


class GuitarDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Guitar Detection System")
        self.root.geometry("1600x900")
        self.root.resizable(True, True)
        
        # Use system default colors
        style = ttk.Style()
        style.theme_use('clam')  # Modern, clean theme
        
        # Use system default background color
        default_bg = style.lookup('TFrame', 'background')
        self.root.configure(bg=default_bg)
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_running = False
        self.preprocessing_enabled = tk.BooleanVar(value=True)
        self.tab_generator = None
        self.detection_thread = None
        self.video_label = None
        self.current_photo = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container with three columns
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill="both", expand=True)
        
        # Left column - controls
        left_frame = ttk.Frame(main_container, width=400)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # Middle column - video display
        middle_frame = ttk.Frame(main_container, width=720)
        middle_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Right column - tab display
        right_frame = ttk.Frame(main_container, width=400)
        right_frame.pack(side="right", fill="both", expand=True)
        
        # === LEFT COLUMN - CONTROLS ===
        # Title
        title_frame = ttk.Frame(left_frame)
        title_frame.pack(pady=(0, 15))
        
        title_label = ttk.Label(
            title_frame,
            text="Guitar Detection System",
            font=("Sans", 14, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Video + Audio + Tab Generation",
            font=("Sans", 9)
        )
        subtitle_label.pack()
        
        # Main content frame
        content_frame = ttk.Frame(left_frame, padding="5")
        content_frame.pack(pady=5, padx=5, fill="both", expand=True)
        
        # Video file section
        video_section = ttk.LabelFrame(
            content_frame,
            text="Video File",
            padding="10"
        )
        video_section.pack(fill="x", pady=(0, 10))
        
        # File path display
        self.path_label = ttk.Label(
            video_section,
            text="No file selected",
            font=("Sans", 9),
            wraplength=480,
            justify="left"
        )
        self.path_label.pack(pady=(0, 8))
        
        # Browse button
        browse_btn = ttk.Button(
            video_section,
            text="Browse...",
            command=self.browse_video
        )
        browse_btn.pack()
        
        # Output section (optional)
        output_section = ttk.LabelFrame(
            content_frame,
            text="Output File (Optional)",
            padding="10"
        )
        output_section.pack(fill="x", pady=(0, 10))
        
        self.output_label = ttk.Label(
            output_section,
            text="No output file (display only)",
            font=("Sans", 9),
            wraplength=480,
            justify="left"
        )
        self.output_label.pack(pady=(0, 8))
        
        output_btn = ttk.Button(
            output_section,
            text="Save As...",
            command=self.browse_output
        )
        output_btn.pack()
        
        # Audio preprocessing section
        preprocess_section = ttk.LabelFrame(
            content_frame,
            text="Audio Preprocessing",
            padding="10"
        )
        preprocess_section.pack(fill="x", pady=(0, 10))
        
        preprocess_check = ttk.Checkbutton(
            preprocess_section,
            text="Enable Audio Preprocessing (Demucs/Spleeter)",
            variable=self.preprocessing_enabled
        )
        preprocess_check.pack(anchor="w")
        
        preprocess_info = ttk.Label(
            preprocess_section,
            text="Removes background noise and vocals for better frequency detection.\n"
                 "This will take a few minutes but greatly improves accuracy.",
            font=("Sans", 8),
            justify="left"
        )
        preprocess_info.pack(anchor="w", pady=(5, 0))
        
        # Button frame for better layout
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=20, fill="x")
        
        # Start button
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Detection",
            command=self.start_detection,
            state="disabled"
        )
        self.start_btn.pack(pady=5, ipady=10, fill="x")
        
        # Status label
        self.status_label = ttk.Label(
            content_frame,
            text="Ready",
            font=("Sans", 9)
        )
        self.status_label.pack(pady=(10, 0))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            content_frame,
            mode='determinate',
            length=500
        )
        self.progress_bar.pack(pady=(5, 0), fill="x")
        self.progress_bar.pack_forget()  # Hide initially
        
        # Instructions
        instructions = ttk.Label(
            left_frame,
            text="1. Select video  •  2. Start detection  •  3. Watch & view tabs",
            font=("Sans", 8)
        )
        instructions.pack(side="bottom", pady=(10, 0))
        
        # === MIDDLE COLUMN - VIDEO DISPLAY ===
        video_title = ttk.Label(
            middle_frame,
            text="Video Preview",
            font=("Sans", 12, "bold")
        )
        video_title.pack(pady=(0, 8))
        
        # Video display frame with border
        video_frame = ttk.Frame(middle_frame, relief="sunken", borderwidth=2)
        video_frame.pack(fill="both", expand=True)
        
        # Video label for displaying frames
        self.video_label = ttk.Label(
            video_frame,
            text="No video playing\n\nSelect a video and start detection to see live preview",
            font=("Sans", 11),
            anchor="center",
            justify="center",
            background="#1a1a1a",
            foreground="#888888"
        )
        self.video_label.pack(fill="both", expand=True)
        
        # === RIGHT COLUMN - TAB DISPLAY ===
        tab_title = ttk.Label(
            right_frame,
            text="Guitar Tablature",
            font=("Sans", 12, "bold")
        )
        tab_title.pack(pady=(0, 8))
        
        # Tab text display frame
        text_frame = ttk.Frame(right_frame, relief="sunken", borderwidth=1)
        text_frame.pack(fill="both", expand=True)
        
        # Tab text display
        self.tab_text = scrolledtext.ScrolledText(
            text_frame,
            font=("Courier", 10),
            bg="#0c0c0c",
            fg="#00ff00",
            insertbackground="#00ff00",
            wrap=tk.NONE,
            height=35,
            relief="flat"
        )
        self.tab_text.pack(fill="both", expand=True)
        self.tab_text.insert("1.0", "No tabs yet. Start detection to generate guitar tabs.\n\n"
                                     "Tabs will appear here in real-time as notes are detected!")
        self.tab_text.config(state=tk.DISABLED)
        
        # Tab controls
        tab_controls = ttk.Frame(right_frame)
        tab_controls.pack(pady=10)
        
        self.save_tab_btn = ttk.Button(
            tab_controls,
            text="Save Tab...",
            command=self.save_tab_file,
            state="disabled"
        )
        self.save_tab_btn.pack(side="left", padx=3)
        
        self.clear_tab_btn = ttk.Button(
            tab_controls,
            text="Clear",
            command=self.clear_tab,
            state="disabled"
        )
        self.clear_tab_btn.pack(side="left", padx=3)
        
    def browse_video(self):
        """Open file dialog to select video file"""
        filetypes = (
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("All files", "*.*")
        )
        
        # Start from home directory to allow browsing anywhere
        filename = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=os.path.expanduser("~"),
            filetypes=filetypes
        )
        
        if filename:
            self.video_path.set(filename)
            # Update display
            display_path = filename
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            self.path_label.config(text=display_path)
            self.start_btn.config(state="normal")
            self.status_label.config(text="Video loaded - Ready to start")
    
    def browse_output(self):
        """Open file dialog to select output location"""
        filetypes = (
            ("MP4 video", "*.mp4"),
            ("AVI video", "*.avi"),
            ("All files", "*.*")
        )
        
        filename = filedialog.asksaveasfilename(
            title="Save Output Video As",
            initialdir=os.path.expanduser("~"),
            filetypes=filetypes,
            defaultextension=".mp4"
        )
        
        if filename:
            self.output_path.set(filename)
            # Update display
            display_path = filename
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            self.output_label.config(text=display_path)
    
    def update_tab_display(self, timestamp, string_num, fret_num):
        """Update tab display when a new note is detected."""
        freq = GUITAR_FREQUENCIES.get((string_num, fret_num), 0)
        note_info = f"[{timestamp:.2f}s] String {string_num}, Fret {fret_num} ({freq:.2f}Hz)\n"
        
        def update():
            self.tab_text.config(state=tk.NORMAL)
            self.tab_text.insert(tk.END, note_info)
            self.tab_text.see(tk.END)
            self.tab_text.config(state=tk.DISABLED)
            self.save_tab_btn.config(state="normal")
            self.clear_tab_btn.config(state="normal")
        
        self.root.after(0, update)
    
    def save_tab_file(self):
        """Save the current tab to a file."""
        if not self.tab_generator or not self.tab_generator.notes:
            messagebox.showwarning("No Tabs", "No tabs to save yet!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Guitar Tab",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.tab_generator.save_to_file(filename)
                messagebox.showinfo("Saved", f"Tab saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save tab:\n{str(e)}")
    
    def clear_tab(self):
        """Clear the tab display."""
        self.tab_text.config(state=tk.NORMAL)
        self.tab_text.delete("1.0", tk.END)
        self.tab_text.insert("1.0", "Tab cleared. Start new detection to generate tabs.\n")
        self.tab_text.config(state=tk.DISABLED)
        self.save_tab_btn.config(state="disabled")
        self.clear_tab_btn.config(state="disabled")
    
    def update_video_frame(self, frame):
        """Update the video display with a new frame (thread-safe)."""
        # Resize frame to fit display (max 720x540)
        display_height = 540
        display_width = int(frame.shape[1] * (display_height / frame.shape[0]))
        if display_width > 720:
            display_width = 720
            display_height = int(frame.shape[0] * (display_width / frame.shape[1]))
        
        frame_resized = cv2.resize(frame, (display_width, display_height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        
        # Convert to PhotoImage (must be done in main thread)
        def update():
            self.current_photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.current_photo, text="")
        
        self.root.after(0, update)
    
    def update_progress(self, status_text, progress_value):
        """Update progress bar and status during detection."""
        self.status_label.config(text=status_text)
        if not self.progress_bar.winfo_viewable():
            self.progress_bar.pack(pady=(5, 0), fill="x", before=self.status_label)
        self.progress_bar['value'] = progress_value
    
    def start_detection(self):
        """Start the detection process"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Detection is already in progress!")
            return
        
        if not self.video_path.get():
            messagebox.showerror("No Video", "Please select a video file first!")
            return
        
        # Check if file exists
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("File Not Found", "The selected video file does not exist!")
            return
        
        # Disable start button
        self.start_btn.config(state="disabled")
        self.status_label.config(text="Starting detection...")
        
        # Run detection in separate thread
        thread = threading.Thread(target=self.run_detection, daemon=True)
        thread.start()
    
    def run_detection(self):
        """Run the detection with embedded tab generation."""
        self.is_running = True
        
        # Initialize tab generator with callback
        self.tab_generator = GuitarTabGenerator(callback=self.update_tab_display)
        
        # Clear previous tab display
        self.root.after(0, lambda: self.tab_text.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.tab_text.delete("1.0", tk.END))
        self.root.after(0, lambda: self.tab_text.insert("1.0", "Detection started...\nWaiting for notes...\n\n"))
        self.root.after(0, lambda: self.tab_text.config(state=tk.DISABLED))
        
        try:
            # Load model
            self.root.after(0, lambda: self.status_label.config(text="Loading model..."))
            model = YOLO(MODEL_PATH)
            print(f"[DEBUG] Model loaded from: {MODEL_PATH}")
            
            # Initialize hand detector
            detector = HandDetector(detectionCon=0.3, minTrackCon=0.3, maxHands=2)
            print(f"[DEBUG] Hand detector initialized (detecting 2 hands)")
            
            # Open video
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[DEBUG] Video opened: {total_frames} frames at {fps} FPS")
            
            # Load and process audio
            has_audio = False
            audio_y, audio_sr = None, None
            
            self.root.after(0, lambda: self.status_label.config(text="Extracting audio..."))
            
            try:
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                
                result = subprocess.run(
                    ['ffmpeg', '-i', self.video_path.get(), '-vn', '-acodec', 'pcm_s16le',
                     '-ar', '44100', '-ac', '1', '-y', temp_audio_path],
                    capture_output=True
                )
                
                if result.returncode == 0:
                    processed_audio_path = temp_audio_path
                    
                    if self.preprocessing_enabled.get():
                        self.root.after(0, lambda: self.status_label.config(
                            text="Preprocessing audio (this may take a while)..."))
                        preprocessor = AudioPreprocessor(method='auto')
                        if preprocessor.is_available():
                            result_path = preprocessor.preprocess_audio(temp_audio_path)
                            if result_path and os.path.exists(result_path):
                                processed_audio_path = result_path
                    
                    # Load audio using scipy (simpler, no librosa needed)
                    audio_sr, audio_data = wavfile.read(processed_audio_path)
                    # Convert to float32 normalized
                    audio_y = audio_data.astype(np.float32) / 32768.0
                    # Ensure mono - if stereo, take first channel or average
                    if len(audio_y.shape) > 1:
                        audio_y = audio_y[:, 0]  # Take first channel
                    has_audio = True
                    print(f"[DEBUG] Audio loaded: {len(audio_y)} samples at {audio_sr}Hz")
                    
                    if processed_audio_path != temp_audio_path:
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass
            except Exception as e:
                print(f"Audio processing warning: {e}")
            
            # Setup video writer if output specified
            video_writer = None
            if self.output_path.get():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter(self.output_path.get(), fourcc, fps, (width, height))
            
            # Detection loop
            self.root.after(0, lambda: self.status_label.config(
                text="Detection running..."))
            self.root.after(0, lambda: self.progress_bar.pack(pady=(5, 0), fill="x", before=self.status_label))
            self.root.after(0, lambda: self.progress_bar.config(value=0))
            
            frame_count = 0
            hand_detection_skip = 0
            fps_counter = 0
            fps_time = time.time()
            fps_display = 0
            show_hands = True
            
            # Debug counters
            debug_audio_detected = 0
            debug_hands_detected = 0
            debug_neck_detected = 0
            debug_notes_added = 0
            
            # Initialize gap tracker for detecting hidden frets
            gap_tracker = FretGapTracker(max_frets=24, stability_frames=8, occlusion_ratio=1.5)
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = frame_count / fps
                
                # Analyze audio
                current_hz, audio_string, audio_fret = None, None, None
                if has_audio:
                    start_sample = int(current_time * audio_sr)
                    end_sample = int((current_time + 1/fps) * audio_sr)
                    if end_sample < len(audio_y):
                        audio_segment = audio_y[start_sample:end_sample]
                        detected_hz, string, fret, exact_hz = analyze_audio_segment(audio_segment, audio_sr)
                        if detected_hz is not None:
                            current_hz, audio_string, audio_fret = detected_hz, string, fret
                            debug_audio_detected += 1
                            if debug_audio_detected <= 5:  # Print first 5 detections
                                print(f"[DEBUG] Audio detected at {current_time:.2f}s: {detected_hz:.1f}Hz -> String {string}, Fret {fret}")
                
                # Process frame
                process_frame = cv2.resize(frame, (640, 480))
                
                # Hand detection
                hand_detection_skip += 1
                hands_list = []
                if hand_detection_skip >= 4:
                    hands_list, _ = detector.findHands(process_frame, draw=False)
                    hand_detection_skip = 0
                    if hands_list and debug_hands_detected <= 5:
                        debug_hands_detected += 1
                        print(f"[DEBUG] Hand detected at frame {frame_count}")
                
                # YOLO detection
                results = model.predict(process_frame, conf=CONFIDENCE_THRESHOLD, 
                                       iou=IOU_THRESHOLD, imgsz=416, verbose=False)
                
                detections = results[0].boxes
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
                neck_coords = None
                nut_coords = None
                detected_frets = []  # List of (fret_number, x1, y1, x2, y2)
                
                # First pass: collect all frets and sort by x position to assign fret numbers
                fret_boxes = []
                
                # Process detections
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1_scaled, x2_scaled = int(x1 * scale_x), int(x2 * scale_x)
                    y1_scaled, y2_scaled = int(y1 * scale_y), int(y2 * scale_y)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
                    color = COLORS.get(class_name, (0, 255, 0))
                    
                    if class_name == 'neck':
                        neck_coords = (x1_scaled, y1_scaled, x2_scaled, y2_scaled)
                        # Draw neck bounding box
                        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                        cv2.putText(frame, f"neck: {conf:.2f}", (x1_scaled, y1_scaled - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        if debug_neck_detected <= 5:
                            debug_neck_detected += 1
                            print(f"[DEBUG] Guitar neck detected at frame {frame_count}: bbox=({x1_scaled},{y1_scaled},{x2_scaled},{y2_scaled})")
                    
                    elif class_name == 'nut':
                        nut_coords = (x1_scaled, y1_scaled, x2_scaled, y2_scaled)
                    
                    elif class_name == 'fret':
                        # Store fret box for sorting
                        fret_boxes.append((x1_scaled, y1_scaled, x2_scaled, y2_scaled, conf))
                
                # Use gap tracker to assign fret numbers intelligently
                # This handles hidden frets by analyzing gap patterns
                nut_center_x = (nut_coords[0] + nut_coords[2]) // 2 if nut_coords else None
                detected_frets, hidden_frets = gap_tracker.assign_fret_numbers(fret_boxes, nut_center_x)
                
                # Combine detected and hidden frets for display
                all_frets = detected_frets + hidden_frets
                all_frets.sort(key=lambda f: f[0])  # Sort by fret number
                
                # Draw detected frets and nut (with any hidden frets restored)
                draw_detected_frets(frame, all_frets, nut_coords)
                
                # Process hands and match with audio
                covered_frets_all = set()
                if show_hands and hands_list and all_frets and neck_coords:
                    for hand in hands_list:
                        lmList = hand["lmList"]
                        
                        # Debug: Check actual landmark positions vs detected frets
                        if frame_count % 60 == 0:  # Every 2 seconds
                            # Get a fingertip (index finger tip = landmark 8)
                            tip_x = int(lmList[8][0] * scale_x)
                            tip_y = int(lmList[8][1] * scale_y)
                            print(f"[DEBUG] Frame {frame_count}: Finger tip at ({tip_x}, {tip_y}), {len(all_frets)} frets ({len(hidden_frets)} restored)")
                        
                        covered_frets = get_frets_from_hand_position(
                            lmList, all_frets, neck_coords, nut_coords, scale_x, scale_y
                        )
                        covered_frets_all.update(covered_frets)
                        
                        # Debug: show what frets are covered
                        if len(covered_frets) > 0 and frame_count % 30 == 0:  # Every 30 frames
                            print(f"[DEBUG] Frame {frame_count}: Hand covering frets: {sorted(covered_frets)}, Audio: {audio_fret if audio_fret else 'None'}")
                        
                        # Check for match with audio
                        if current_hz and audio_string and audio_fret:
                            if audio_fret in covered_frets:
                                self.tab_generator.add_note(current_time, audio_string, audio_fret)
                                debug_notes_added += 1
                                if debug_notes_added <= 5:
                                    print(f"[DEBUG] ✓ NOTE ADDED: {current_time:.2f}s -> String {audio_string}, Fret {audio_fret} ({current_hz:.1f}Hz)")
                            else:
                                for fret in covered_frets:
                                    for string in range(1, 7):
                                        if (string, fret) in GUITAR_FREQUENCIES:
                                            freq = GUITAR_FREQUENCIES[(string, fret)]
                                            if abs(freq - current_hz) <= 5.0:
                                                self.tab_generator.add_note(current_time, string, fret)
                                                debug_notes_added += 1
                                                if debug_notes_added <= 5:
                                                    print(f"[DEBUG] ✓ NOTE ADDED (alt): {current_time:.2f}s -> String {string}, Fret {fret} ({freq:.1f}Hz)")
                                                break
                
                # FPS calculation
                fps_counter += 1
                if time.time() - fps_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                # Draw info on frame (for output video if enabled)
                cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_hz:
                    cv2.putText(frame, f"Audio: {current_hz:.1f}Hz - S{audio_string}F{audio_fret}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if video_writer:
                    video_writer.write(frame)
                
                # Update video display in GUI (every frame for smooth playback)
                self.update_video_frame(frame)
                
                # Update progress in GUI (every 30 frames to avoid flooding)
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    status_text = f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)"
                    if len(self.tab_generator.notes) > 0:
                        status_text += f" | {len(self.tab_generator.notes)} notes detected"
                    self.root.after(0, lambda t=status_text, p=progress: self.update_progress(t, p))
            
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            
            # Print debug summary
            print("\n" + "="*60)
            print("DETECTION SUMMARY")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            print(f"Audio frequencies detected: {debug_audio_detected}")
            print(f"Hands detected: {debug_hands_detected}")
            print(f"Guitar necks detected: {debug_neck_detected}")
            print(f"Notes added to tab: {debug_notes_added}")
            print(f"Total unique notes: {len(self.tab_generator.notes)}")
            print("="*60 + "\n")
            
            # Display final tab
            if self.tab_generator.notes:
                final_tab = f"\n\n{'='*60}\nFINAL TAB\n{'='*60}\n\n"
                final_tab += self.tab_generator.generate_tab()
                
                self.root.after(0, lambda: self.tab_text.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.tab_text.insert(tk.END, final_tab))
                self.root.after(0, lambda: self.tab_text.see(tk.END))
                self.root.after(0, lambda: self.tab_text.config(state=tk.DISABLED))
            
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.config(
                text="Detection completed successfully!"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Complete", f"Detection completed!\nTotal notes detected: {len(self.tab_generator.notes)}"))
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))
        
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

def main():
    root = tk.Tk()
    app = GuitarDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
