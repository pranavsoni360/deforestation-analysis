#!/usr/bin/env python3
"""
Forest Area Analysis - Quick Demo Launcher

This script launches the forest analysis system in demo mode without requiring
database setup or complex data fetching. Perfect for demonstrations and testing.
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

def print_banner():
    """Print a nice banner for the demo."""
    print("🌳" * 20)
    print("FOREST AREA ANALYSIS SYSTEM")
    print("Quick Demo Mode")
    print("🌳" * 20)
    print()

def check_dependencies():
    """Check if required packages are available."""
    required_packages = ['flask', 'numpy', 'pandas']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages found")
    return True

def open_browser_after_delay():
    """Open the browser after a short delay."""
    time.sleep(3)
    print("\n🌐 Opening web interface...")
    webbrowser.open('http://localhost:5000/simple')

def main():
    print_banner()
    
    print("🔍 Checking system requirements...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting Forest Analysis Demo...")
    print("   This demo includes:")
    print("   ✅ Interactive web interface")
    print("   ✅ Real-time forest analysis")
    print("   ✅ Global location support")
    print("   ✅ Instant sustainability metrics")
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser_after_delay)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n📡 Starting web server...")
    print("   Web Interface: http://localhost:5000/simple")
    print("   API Endpoint: http://localhost:5000/api/analyze-simple")
    print()
    print("💡 Try these demo features:")
    print("   • Click on map locations")
    print("   • Use example coordinates")
    print("   • Test different forest types")
    print("   • Download analysis reports")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run Flask app
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import the Flask app
        from app import app
        
        # Start the server
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped. Thank you for trying Forest Area Analysis!")
        print("\nFor more features, try:")
        print("  • python simple_demo.py --interactive  (Terminal demo)")
        print("  • python simple_demo.py                (Scenario comparison)")
    except Exception as e:
        print(f"\n❌ Error starting demo: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure port 5000 is not in use")
        print("  • Check that all files are in place")
        print("  • Try: python simple_demo.py --interactive")

if __name__ == "__main__":
    main()
