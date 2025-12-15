#!/usr/bin/env python3
"""
Guitar Fret Detector
Analyzes audio files and maps detected frequencies to guitar string and fret positions
"""

import librosa
import numpy as np
import sys
from pathlib import Path


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


def analyze_guitar_audio(audio_path, window_duration=0.1):
    """
    Analyze audio file and map frequencies to guitar fret positions.
    
    Args:
        audio_path: Path to the audio file
        window_duration: Duration of each analysis window in seconds (default: 0.1)
    
    Returns:
        List of detected fret positions at each time interval
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    print(f"\n{'='*70}")
    print(f"Audio File: {Path(audio_path).name}")
    print(f"{'='*70}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr:,} Hz")
    print(f"Analysis window: {window_duration} seconds")
    print(f"\n{'='*70}")
    print(f"Guitar Fret Detection (every {window_duration}s):")
    print(f"{'='*70}\n")
    
    window_samples = int(sr * window_duration)
    hop_samples = window_samples  # Non-overlapping windows
    
    results = []
    window_num = 0
    
    for i in range(0, len(y) - window_samples, hop_samples):
        window = y[i:i + window_samples]
        start_time = i / sr
        end_time = (i + window_samples) / sr
        window_num += 1
        
        # Extract pitch for this window
        f0, voiced_flag, voiced_probs = librosa.pyin(
            window, 
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
                print(f"[{start_time:5.2f}s - {end_time:5.2f}s]  {detected_hz:7.2f} Hz  →  String {string}, Fret {fret:2d}  ({exact_hz:.2f} Hz)")
                
                results.append({
                    'window': window_num,
                    'start_time': start_time,
                    'end_time': end_time,
                    'detected_hz': detected_hz,
                    'string': string,
                    'fret': fret,
                    'exact_hz': exact_hz
                })
            else:
                print(f"[{start_time:5.2f}s - {end_time:5.2f}s]  {detected_hz:7.2f} Hz  →  (no data)")
                
                results.append({
                    'window': window_num,
                    'start_time': start_time,
                    'end_time': end_time,
                    'detected_hz': detected_hz,
                    'string': None,
                    'fret': None,
                    'exact_hz': None
                })
        else:
            print(f"[{start_time:5.2f}s - {end_time:5.2f}s]  ------  (no pitch detected)")
            
            results.append({
                'window': window_num,
                'start_time': start_time,
                'end_time': end_time,
                'detected_hz': None,
                'string': None,
                'fret': None,
                'exact_hz': None
            })
    
    print(f"\n{'='*70}")
    print(f"Total windows analyzed: {window_num}")
    print(f"{'='*70}\n")
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python guitar_fret_detector.py <audio_file> [window_duration]")
        print("\nExample:")
        print("  python guitar_fret_detector.py f1s1.mp3")
        print("  python guitar_fret_detector.py f1s1.mp3 0.1")
        print("  python guitar_fret_detector.py f1s1.mp3 0.2")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    window_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    
    if not Path(audio_file).exists():
        print(f"Error: File '{audio_file}' not found")
        sys.exit(1)
    
    # Analyze the audio file
    results = analyze_guitar_audio(audio_file, window_duration)


if __name__ == "__main__":
    main()
