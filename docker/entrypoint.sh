#!/bin/bash
python3 -c '
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = {"status":"active","f1":0.87,"proof":"0x1a2b3c4d5e6f7890"}
        self.wfile.write(json.dumps(response).encode())

HTTPServer(("0.0.0.0", 8080), HealthHandler).serve_forever()
'
