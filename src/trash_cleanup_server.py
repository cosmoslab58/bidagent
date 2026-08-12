#!/usr/bin/env python3
"""Simple HTTP server for trash cleanup. No Node-RED dashboard needed."""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = os.environ.get("CC_TOKEN", "***")
HOST = "curbclass.localhost:3000" 
BASE = "http://bodhi.lab:3100"
TYPES = ["people", "companies", "opportunities", "notes", "tasks", "attachments"]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        workspace = self.path.strip("/").split("/")[-1]
        if workspace not in ("curbclass", "chassisshield"):
            self.send_error(400, json.dumps({"error": "Invalid workspace"}))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        results = {}
        for obj_type in TYPES:
            deleted = 0
            failed = 0
            page = 0
            while page < 50:
                search_url = f"{BASE}/rest/{obj_type}?filter=deletedAt[is]:NOT_NULL&limit=60"
                search_req = urllib.request.Request(search_url, headers={"Authorization": f"Bearer {TOKEN}", "Host": HOST})
                try:
                    resp = urllib.request.urlopen(search_req, timeout=10)
                    data = json.loads(resp.read())
                    records = data.get("data", {}).get(obj_type, [])
                except Exception:
                    break
                if not records:
                    break
                for record in records:
                    del_url = f"{BASE}/rest/{obj_type}/{record['id']}"
                    del_req = urllib.request.Request(del_url, method="DELETE", headers={"Authorization": f"Bearer {TOKEN}", "Host": HOST})
                    try:
                        urllib.request.urlopen(del_req, timeout=10)
                        deleted += 1
                    except Exception:
                        failed += 1
                page += 1
            results[obj_type] = {"deleted": deleted, "failed": failed}
        results["totalDeleted"] = sum(v.get("deleted", 0) for v in results.values() if isinstance(v, dict))
        results["totalFailed"] = sum(v.get("failed", 0) for v in results.values() if isinstance(v, dict))
        results["workspace"] = workspace
        self.wfile.write(json.dumps(results, indent=2).encode())

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8343
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving trash cleanup on port {port}", flush=True)
    print("  POST /trash-cleanup/curbclass", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
