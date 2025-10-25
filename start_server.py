#!/usr/bin/env python3
# Dedicated server starter without auto-reloader or debug
# Ensures stable startup for the one-command runner on Windows

from app import app

if __name__ == "__main__":
    # Production-like single-process run (sufficient for local use)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
