"""
Fun Guitar API - Note & Frequency Lookup
Vercel Serverless Function

Provides guitar frequency data, note identification, and tuning information.
"""

from http.server import BaseHTTPRequestHandler
import json
import math
from urllib.parse import urlparse, parse_qs


# Standard guitar tuning
STANDARD_TUNING = {
    6: {'note': 'E2', 'freq': 82.41},
    5: {'note': 'A2', 'freq': 110.00},
    4: {'note': 'D3', 'freq': 146.83},
    3: {'note': 'G3', 'freq': 196.00},
    2: {'note': 'B3', 'freq': 246.94},
    1: {'note': 'E4', 'freq': 329.63},
}

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def freq_to_note(frequency):
    """Convert a frequency to the closest note name and cents deviation."""
    if frequency <= 0:
        return None, None, None
    
    # Calculate semitones from A4 (440 Hz)
    semitones = 12 * math.log2(frequency / 440.0)
    rounded_semitones = round(semitones)
    cents = (semitones - rounded_semitones) * 100
    
    # Calculate note index and octave
    note_index = (rounded_semitones + 9) % 12  # A is index 9 from C
    octave = 4 + (rounded_semitones + 9) // 12
    
    note_name = NOTE_NAMES[note_index] + str(octave)
    exact_freq = 440.0 * (2 ** (rounded_semitones / 12))
    
    return note_name, exact_freq, round(cents, 1)


def get_fret_frequency(string_num, fret_num):
    """Calculate frequency for a given string and fret using 12-TET formula."""
    if string_num not in STANDARD_TUNING:
        return None
    
    open_freq = STANDARD_TUNING[string_num]['freq']
    return open_freq * (2 ** (fret_num / 12))


def generate_fretboard_data(max_fret=20):
    """Generate complete fretboard frequency data."""
    fretboard = {}
    for string in range(1, 7):
        fretboard[str(string)] = {}
        for fret in range(0, max_fret + 1):
            freq = get_fret_frequency(string, fret)
            note, _, _ = freq_to_note(freq)
            fretboard[str(string)][str(fret)] = {
                'frequency': round(freq, 2),
                'note': note
            }
    return fretboard


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        action = params.get('action', ['info'])[0]
        
        if action == 'identify':
            # Identify a frequency -> note
            freq = float(params.get('freq', [0])[0])
            if freq > 0:
                note, exact_freq, cents = freq_to_note(freq)
                response = {
                    'detected_frequency': freq,
                    'closest_note': note,
                    'note_frequency': exact_freq,
                    'cents_deviation': cents,
                    'in_tune': abs(cents) <= 5
                }
            else:
                response = {'error': 'Provide ?freq=<hz> parameter'}
        
        elif action == 'fret':
            # Get frequency for string/fret
            string = int(params.get('string', [1])[0])
            fret = int(params.get('fret', [0])[0])
            freq = get_fret_frequency(string, fret)
            if freq:
                note, _, _ = freq_to_note(freq)
                response = {
                    'string': string,
                    'fret': fret,
                    'frequency': round(freq, 2),
                    'note': note
                }
            else:
                response = {'error': 'Invalid string number (1-6)'}
        
        elif action == 'fretboard':
            # Full fretboard data
            max_fret = int(params.get('max_fret', [20])[0])
            response = {
                'tuning': 'standard',
                'fretboard': generate_fretboard_data(min(max_fret, 24))
            }
        
        elif action == 'tuning':
            # Standard tuning reference
            response = {
                'tuning': 'standard',
                'strings': {
                    str(k): v for k, v in STANDARD_TUNING.items()
                }
            }
        
        else:
            response = {
                'endpoints': {
                    'identify': '/api/notes?action=identify&freq=440',
                    'fret': '/api/notes?action=fret&string=1&fret=0',
                    'fretboard': '/api/notes?action=fretboard&max_fret=20',
                    'tuning': '/api/notes?action=tuning'
                }
            }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
