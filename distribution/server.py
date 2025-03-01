import http.server
import socketserver

PORT = 8000

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = http.server.SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        '.rar': 'application/x-rar-compressed',
        '.zip': 'application/zip',
    })

import os
os.chdir('public')

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Serving at http://{httpd.server_address[0]}:{PORT}")
    httpd.serve_forever()
