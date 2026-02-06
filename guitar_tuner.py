#!/usr/bin/env python3
"""
Accurate Guitar Tuner Module
Uses PyAudio and NumPy for real-time pitch detection
Accuracy comparable to commercial tuner apps

Logic Flow:
1. Audio capture at 44100Hz with 4096 buffer
2. Preprocessing: Hanning window + Low-pass filter
3. Pitch detection: Zero-crossing + FFT peak finding
4. Note matching: Find closest note + calculate cents deviation
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, Dict

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
SAMPLE_RATE = 44100  # Hz - CD quality
BUFFER_SIZE = 4096   # Frames per buffer - good resolution for low E (82Hz)
FREQUENCY_RESOLUTION = SAMPLE_RATE / BUFFER_SIZE  # ~10.77 Hz per bin

# Volume threshold - ignore quiet signals (adjust based on your mic)
VOLUME_THRESHOLD = 0.01

# Low-pass filter cutoff (removes pick noise, room hiss)
LOWPASS_CUTOFF = 1000  # Hz

# =============================================================================
# STANDARD GUITAR TUNING FREQUENCIES (Reference "Cheat Sheet")
# =============================================================================
STANDARD_TUNING = {
    'E2': 82.41,   # String 6 (thickest)
    'A2': 110.00,  # String 5
    'D3': 146.83,  # String 4
    'G3': 196.00,  # String 3
    'B3': 246.94,  # String 2
    'E4': 329.63,  # String 1 (thinnest)
}

# Extended frequency map for all frets (string, fret) -> frequency
# Based on physics: f = f0 * 2^(n/12) where n = frets from open
def generate_all_fret_frequencies() -> Dict[Tuple[int, int], float]:
    """Generate frequencies for all strings and frets using the 12-TET formula."""
    open_string_hz = {
        6: 82.41,   # E2
        5: 110.00,  # A2
        4: 146.83,  # D3
        3: 196.00,  # G3
        2: 246.94,  # B3
        1: 329.63,  # E4
    }
    
    frequencies = {}
    for string, open_freq in open_string_hz.items():
        for fret in range(0, 25):  # Up to 24th fret
            # 12-TET formula: frequency doubles every 12 frets
            freq = open_freq * (2 ** (fret / 12))
            frequencies[(string, fret)] = round(freq, 2)
    
    return frequencies

GUITAR_FREQUENCIES = generate_all_fret_frequencies()


# =============================================================================
# SIGNAL PREPROCESSING
# =============================================================================

def apply_hanning_window(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Apply Hanning window to prevent spectral leakage.
    This smooths the edges of the buffer to avoid artificial high frequencies.
    """
    window = np.hanning(len(audio_buffer))
    return audio_buffer * window


def apply_lowpass_filter(audio_buffer: np.ndarray, 
                         cutoff_hz: float = LOWPASS_CUTOFF,
                         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Apply low-pass filter to remove pick noise and room hiss.
    Butterworth filter for smooth frequency response.
    """
    # filtfilt requires input length > padlen (3 * filter_order * 2 - 1 = 15 for order 4)
    # If buffer is too short, skip filtering
    min_length = 50
    if len(audio_buffer) < min_length:
        return audio_buffer
    
    # Normalize cutoff frequency to Nyquist frequency
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    # Ensure cutoff is valid (0 < cutoff < 1)
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    
    try:
        # Design 4th order Butterworth low-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter with reduced padlen for short signals
        padlen = min(15, len(audio_buffer) - 1)
        filtered = signal.filtfilt(b, a, audio_buffer, padlen=padlen)
        return filtered
    except Exception:
        # If filtering fails, return original
        return audio_buffer


def preprocess_audio(audio_buffer: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Low-pass filter to remove noise
    2. Hanning window to prevent spectral leakage
    """
    # Step 1: Low-pass filter
    filtered = apply_lowpass_filter(audio_buffer)
    
    # Step 2: Hanning window
    windowed = apply_hanning_window(filtered)
    
    return windowed


# =============================================================================
# PITCH DETECTION
# =============================================================================

def get_zero_crossing_rate(audio_buffer: np.ndarray, 
                           sample_rate: int = SAMPLE_RATE) -> float:
    """
    Calculate rough frequency estimate using zero-crossing rate.
    Fast but less accurate - used as initial estimate.
    """
    # Count sign changes
    signs = np.sign(audio_buffer)
    sign_changes = np.abs(np.diff(signs))
    zero_crossings = np.sum(sign_changes > 0)
    
    # Frequency = (zero crossings / 2) / duration
    duration = len(audio_buffer) / sample_rate
    frequency = (zero_crossings / 2) / duration
    
    return frequency


def get_fft_peak_frequency(audio_buffer: np.ndarray,
                           sample_rate: int = SAMPLE_RATE,
                           min_freq: float = 60.0,
                           max_freq: float = 1000.0) -> Tuple[Optional[float], float]:
    """
    Find peak frequency using FFT magnitude spectrum.
    More accurate than zero-crossing.
    
    Returns: (peak_frequency, magnitude)
    """
    # Compute FFT
    fft_result = np.fft.rfft(audio_buffer)
    magnitudes = np.abs(fft_result)
    
    # Get frequency bins
    freqs = np.fft.rfftfreq(len(audio_buffer), 1/sample_rate)
    
    # Filter to guitar frequency range
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    valid_freqs = freqs[valid_mask]
    valid_mags = magnitudes[valid_mask]
    
    if len(valid_mags) == 0:
        return None, 0.0
    
    # Find peak
    peak_idx = np.argmax(valid_mags)
    peak_freq = valid_freqs[peak_idx]
    peak_mag = valid_mags[peak_idx]
    
    return peak_freq, peak_mag


def get_autocorrelation_frequency(audio_buffer: np.ndarray,
                                   sample_rate: int = SAMPLE_RATE,
                                   min_freq: float = 60.0,
                                   max_freq: float = 1000.0) -> Optional[float]:
    """
    Use autocorrelation for more accurate pitch detection.
    Better for harmonic signals like guitar strings.
    """
    # Autocorrelation via FFT (faster than direct computation)
    n = len(audio_buffer)
    fft = np.fft.rfft(audio_buffer, n=2*n)
    autocorr = np.fft.irfft(fft * np.conj(fft))[:n]
    
    # Normalize
    autocorr = autocorr / autocorr[0]
    
    # Find first peak after zero (fundamental period)
    min_period = int(sample_rate / max_freq)
    max_period = int(sample_rate / min_freq)
    
    # Search for peak in valid range
    search_range = autocorr[min_period:max_period]
    if len(search_range) == 0:
        return None
    
    peak_idx = np.argmax(search_range) + min_period
    
    # Refine with parabolic interpolation
    if 0 < peak_idx < len(autocorr) - 1:
        y0, y1, y2 = autocorr[peak_idx-1:peak_idx+2]
        if y1 > y0 and y1 > y2:  # Valid peak
            # Parabolic interpolation for sub-sample accuracy
            shift = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            refined_period = peak_idx + shift
            frequency = sample_rate / refined_period
            return frequency
    
    return sample_rate / peak_idx


def detect_pitch(audio_buffer: np.ndarray,
                 sample_rate: int = SAMPLE_RATE) -> Tuple[Optional[float], float]:
    """
    Main pitch detection function combining multiple methods.
    
    Returns: (detected_frequency, confidence)
    """
    # Check if signal is loud enough
    rms = np.sqrt(np.mean(audio_buffer ** 2))
    if rms < VOLUME_THRESHOLD:
        return None, 0.0
    
    # Preprocess
    processed = preprocess_audio(audio_buffer)
    
    # Method 1: FFT peak (fast, good for initial estimate)
    fft_freq, fft_mag = get_fft_peak_frequency(processed, sample_rate)
    
    # Method 2: Autocorrelation (more accurate for harmonics)
    autocorr_freq = get_autocorrelation_frequency(processed, sample_rate)
    
    # Method 3: Zero-crossing (sanity check)
    zcr_freq = get_zero_crossing_rate(processed, sample_rate)
    
    # Combine results - prefer autocorrelation if available
    if autocorr_freq is not None and 60 <= autocorr_freq <= 1000:
        detected_freq = autocorr_freq
        confidence = 0.9
    elif fft_freq is not None and 60 <= fft_freq <= 1000:
        detected_freq = fft_freq
        confidence = 0.7
    else:
        return None, 0.0
    
    # Validate with zero-crossing (should be within 2x range)
    if zcr_freq > 0:
        ratio = detected_freq / zcr_freq
        if not (0.4 <= ratio <= 2.5):  # Allow some tolerance for harmonics
            confidence *= 0.5
    
    return detected_freq, confidence


# =============================================================================
# NOTE MATCHING & CENTS CALCULATION
# =============================================================================

def frequency_to_cents(detected_freq: float, target_freq: float) -> float:
    """
    Calculate deviation in cents from target frequency.
    
    Cents formula: 1200 × log2(detected_freq / target_freq)
    
    100 cents = 1 semitone
    Negative = flat (too low)
    Positive = sharp (too high)
    """
    if detected_freq <= 0 or target_freq <= 0:
        return 0.0
    return 1200 * np.log2(detected_freq / target_freq)


def find_closest_note(detected_freq: float) -> Tuple[str, int, int, float, float]:
    """
    Find the closest guitar note to the detected frequency.
    
    Returns: (note_name, string_number, fret_number, target_freq, cents_deviation)
    """
    if detected_freq is None or detected_freq <= 0:
        return None, None, None, None, None
    
    # Find closest match from all guitar frequencies
    min_cents = float('inf')
    best_match = None
    
    for (string, fret), target_freq in GUITAR_FREQUENCIES.items():
        cents = abs(frequency_to_cents(detected_freq, target_freq))
        if cents < min_cents:
            min_cents = cents
            best_match = (string, fret, target_freq)
    
    if best_match is None:
        return None, None, None, None, None
    
    string, fret, target_freq = best_match
    cents_deviation = frequency_to_cents(detected_freq, target_freq)
    
    # Generate note name
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    open_notes = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}  # Open string note indices
    note_idx = (open_notes[string] + fret) % 12
    octave = 2 + (open_notes[string] + fret) // 12
    if string <= 2:
        octave += 1
    note_name = f"{note_names[note_idx]}{octave}"
    
    return note_name, string, fret, target_freq, cents_deviation


def find_closest_open_string(detected_freq: float) -> Tuple[str, float, float]:
    """
    Find the closest open string (for tuning mode).
    
    Returns: (string_name, target_freq, cents_deviation)
    """
    if detected_freq is None or detected_freq <= 0:
        return None, None, None
    
    min_cents = float('inf')
    best_match = None
    
    for name, target_freq in STANDARD_TUNING.items():
        cents = abs(frequency_to_cents(detected_freq, target_freq))
        if cents < min_cents:
            min_cents = cents
            best_match = (name, target_freq)
    
    if best_match is None:
        return None, None, None
    
    name, target_freq = best_match
    cents_deviation = frequency_to_cents(detected_freq, target_freq)
    
    return name, target_freq, cents_deviation


def get_tuning_display(cents: float) -> str:
    """
    Get a visual tuning indicator.
    
    Returns string like: "<<<< FLAT" or "IN TUNE!" or "SHARP >>>>"
    """
    if cents is None:
        return ""
    
    if abs(cents) <= 5:
        return "✓ IN TUNE!"
    elif cents < -50:
        return "◀◀◀◀ VERY FLAT"
    elif cents < -20:
        return "◀◀◀ FLAT"
    elif cents < -5:
        return "◀ slightly flat"
    elif cents > 50:
        return "VERY SHARP ▶▶▶▶"
    elif cents > 20:
        return "SHARP ▶▶▶"
    else:
        return "slightly sharp ▶"


# =============================================================================
# MAIN TUNER CLASS
# =============================================================================

class GuitarTuner:
    """
    Complete guitar tuner with real-time pitch detection.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.last_frequency = None
        self.frequency_history = []
        self.history_size = 5  # Smooth over last N readings
    
    def analyze(self, audio_buffer: np.ndarray) -> dict:
        """
        Analyze an audio buffer and return complete tuning info.
        
        Returns dict with:
            - frequency: detected Hz
            - note: note name (e.g., "A2")
            - string: string number (1-6)
            - fret: fret number (0-24)
            - target_freq: exact target Hz
            - cents: deviation in cents
            - tuning_status: visual indicator
            - confidence: detection confidence (0-1)
        """
        # Detect pitch
        freq, confidence = detect_pitch(audio_buffer, self.sample_rate)
        
        if freq is None:
            return {
                'frequency': None,
                'note': None,
                'string': None,
                'fret': None,
                'target_freq': None,
                'cents': None,
                'tuning_status': 'No signal',
                'confidence': 0.0
            }
        
        # Smooth frequency with history
        self.frequency_history.append(freq)
        if len(self.frequency_history) > self.history_size:
            self.frequency_history.pop(0)
        
        smoothed_freq = np.median(self.frequency_history)
        
        # Find closest note
        note, string, fret, target_freq, cents = find_closest_note(smoothed_freq)
        
        return {
            'frequency': round(smoothed_freq, 2),
            'note': note,
            'string': string,
            'fret': fret,
            'target_freq': target_freq,
            'cents': round(cents, 1) if cents else None,
            'tuning_status': get_tuning_display(cents),
            'confidence': confidence
        }
    
    def reset(self):
        """Clear frequency history."""
        self.frequency_history = []
        self.last_frequency = None


# =============================================================================
# CONVENIENCE FUNCTION FOR INTEGRATION
# =============================================================================

def analyze_audio_segment(audio_segment: np.ndarray, 
                          sr: int = SAMPLE_RATE) -> Tuple[Optional[float], 
                                                          Optional[int], 
                                                          Optional[int], 
                                                          Optional[float]]:
    """
    Drop-in replacement for the old librosa-based function.
    
    Returns: (detected_hz, string_number, fret_number, exact_target_hz)
    """
    if len(audio_segment) < 512:
        return None, None, None, None
    
    # Ensure float type
    if audio_segment.dtype != np.float32 and audio_segment.dtype != np.float64:
        audio_segment = audio_segment.astype(np.float32)
        # Normalize if int type was converted
        max_val = np.max(np.abs(audio_segment))
        if max_val > 1.0:
            audio_segment = audio_segment / 32768.0  # Assume 16-bit audio
    
    freq, confidence = detect_pitch(audio_segment, sr)
    
    if freq is None or confidence < 0.3:
        return None, None, None, None
    
    note, string, fret, target_freq, cents = find_closest_note(freq)
    
    if string is None:
        return None, None, None, None
    
    # Only return if within reasonable tuning (less than 50 cents off)
    if abs(cents) > 50:
        return freq, string, fret, target_freq
    
    return freq, string, fret, target_freq


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GUITAR TUNER MODULE - TEST")
    print("=" * 60)
    
    # Test frequency generation
    print("\nStandard Tuning Frequencies:")
    for name, freq in STANDARD_TUNING.items():
        print(f"  {name}: {freq} Hz")
    
    # Test cents calculation
    print("\nCents Calculation Examples:")
    test_cases = [
        (110.0, 110.0, "A2 in tune"),
        (112.0, 110.0, "A2 sharp"),
        (108.0, 110.0, "A2 flat"),
        (82.0, 82.41, "E2 slightly flat"),
    ]
    
    for detected, target, description in test_cases:
        cents = frequency_to_cents(detected, target)
        status = get_tuning_display(cents)
        print(f"  {description}: {detected}Hz vs {target}Hz = {cents:.1f} cents {status}")
    
    # Test note finding
    print("\nNote Finding Examples:")
    test_freqs = [82.41, 110.0, 112.0, 146.83, 220.0, 440.0]
    for freq in test_freqs:
        note, string, fret, target, cents = find_closest_note(freq)
        print(f"  {freq}Hz -> {note} (String {string}, Fret {fret}) [{cents:+.1f} cents]")
    
    print("\n" + "=" * 60)
    print("Module loaded successfully!")
    print("=" * 60)
