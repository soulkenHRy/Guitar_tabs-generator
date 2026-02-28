"""
Fun Guitar API - Tab Formatter
Vercel Serverless Function

Converts note data into formatted guitar tablature.
"""

from http.server import BaseHTTPRequestHandler
import json


# Tab formatting constants
TAB_WIDTH = 60  # Characters per line
STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E']  # High to low


def format_tab(notes, measures_per_line=4):
    """
    Format a list of notes into guitar tablature text.
    
    Args:
        notes: List of dicts with 'string' (1-6) and 'fret' (0-24) keys
        measures_per_line: How many measures per tab line
    
    Returns:
        Formatted tab string
    """
    if not notes:
        return "No notes to display."
    
    # Initialize strings (6 strings)
    strings = {i: '' for i in range(1, 7)}
    
    # Track position for alignment
    for note in notes:
        string_num = note.get('string', 1)
        fret_num = note.get('fret', 0)
        
        if string_num < 1 or string_num > 6:
            continue
        
        fret_str = str(fret_num)
        pad_len = len(fret_str)
        
        # Add the note to the correct string, dashes to others
        for s in range(1, 7):
            if s == string_num:
                strings[s] += fret_str + '-'
            else:
                strings[s] += '-' * pad_len + '-'
    
    # Add trailing dashes to make all strings equal length
    max_len = max(len(s) for s in strings.values())
    for s in range(1, 7):
        strings[s] = strings[s].ljust(max_len, '-')
    
    # Split into lines of TAB_WIDTH
    result_lines = []
    pos = 0
    
    while pos < max_len:
        end = min(pos + TAB_WIDTH, max_len)
        
        for i, name in enumerate(STRING_NAMES):
            string_num = i + 1  # 1=high E, 6=low E ... but convention is:
            # STRING_NAMES[0] = 'e' = string 1 (high E)
            line = f"{name}|{strings[string_num][pos:end]}|"
            result_lines.append(line)
        
        result_lines.append('')  # Blank line between sections
        pos = end
    
    return '\n'.join(result_lines)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            data = json.loads(body) if body else {}
            notes = data.get('notes', [])
            
            if not notes:
                response = {
                    'error': 'No notes provided',
                    'usage': {
                        'method': 'POST',
                        'body': {
                            'notes': [
                                {'string': 1, 'fret': 0},
                                {'string': 2, 'fret': 1},
                                {'string': 3, 'fret': 0}
                            ]
                        }
                    }
                }
            else:
                tab_text = format_tab(notes)
                response = {
                    'tab': tab_text,
                    'note_count': len(notes)
                }
        except Exception as e:
            response = {'error': str(e)}
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Demo tab
        demo_notes = [
            {'string': 1, 'fret': 0}, {'string': 1, 'fret': 2},
            {'string': 1, 'fret': 3}, {'string': 2, 'fret': 1},
            {'string': 2, 'fret': 3}, {'string': 3, 'fret': 0},
            {'string': 3, 'fret': 2}, {'string': 4, 'fret': 0},
            {'string': 4, 'fret': 2}, {'string': 5, 'fret': 0},
            {'string': 5, 'fret': 2}, {'string': 5, 'fret': 3},
        ]
        
        response = {
            'demo': True,
            'tab': format_tab(demo_notes),
            'note_count': len(demo_notes),
            'usage': 'POST notes array to generate custom tabs'
        }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
