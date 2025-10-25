#!/usr/bin/env python3
"""
🌳 Forest Analysis System - Single Command Launcher
==================================================

This is your ONE-COMMAND solution to run the entire Forest Analysis System.
Just run: python run.py

Features:
- Automatic dependency checking and installation
- Clean server startup
- Opens browser automatically
- Works on Windows, Mac, Linux
- No complex setup required
"""

import os
import sys
import time
import webbrowser
import threading
import subprocess
from pathlib import Path

def print_banner():
    """Print the system banner."""
    print("🌳" * 50)
    print("FOREST ANALYSIS SYSTEM")
    print("Advanced Environmental Monitoring Platform")
    print("🌳" * 50)
    print()

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'flask_cors', 'numpy', 'pandas', 'reportlab'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are available")
    
    return True

def kill_existing_server():
    """Kill any existing server on port 5000."""
    try:
        # Find processes using port 5000
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if ':5000' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    try:
                        subprocess.run(['taskkill', '/PID', pid, '/F'], 
                                     capture_output=True, check=False)
                        print(f"🛑 Stopped existing server (PID: {pid})")
                    except:
                        pass
    except:
        pass

def open_browser():
    """Open browser after server starts."""
    time.sleep(3)
    print("🌐 Opening web interface...")
    webbrowser.open('http://localhost:5000/interface')

def main():
    """Main launcher function."""
    print_banner()
    
    # Check dependencies
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Kill existing server
    kill_existing_server()
    
    print("🚀 Starting Forest Analysis System...")
    print("   Features included:")
    print("   ✅ Real-time forest analysis")
    print("   ✅ Interactive mapping interface")
    print("   ✅ Sustainability metrics")
    print("   ✅ PDF report generation")
    print("   ✅ Global location support")
    print("   ✅ Advanced dashboard")
    print()
    
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("📡 Starting web server...")
    print("   Main Interface: http://localhost:5000/interface")
    print("   Dashboard: http://localhost:5000/dashboard")
    print("   API Root: http://localhost:5000/")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n👋 Forest Analysis System stopped. Thank you for using our platform!")
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure port 5000 is not in use")
        print("  • Check that all files are in place")
        print("  • Try: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
