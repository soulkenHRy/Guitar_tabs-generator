"""
Fun Guitar API - Health Check
Vercel Serverless Function
"""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "status": "ok",
            "service": "Fun Guitar API",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "features": {
                "tuner": True,
                "note_lookup": True,
                "tab_format": True,
                "frequency_analysis": True
            }
        }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
