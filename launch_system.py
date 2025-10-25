#!/usr/bin/env python3
"""
Launch System for Deforestation Risk Analysis

This script orchestrates the complete system launch including:
- Database initialization
- Data fetching (optional)
- ML model training (optional)
- Web server startup
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import webbrowser
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print system banner."""
    print("🌳" * 25)
    print("DEFORESTATION RISK ANALYSIS SYSTEM")
    print("Complete Environmental Monitoring Platform")
    print("🌳" * 25)
    print()

def check_python_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        'flask', 'pandas', 'numpy', 'sqlalchemy', 'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"Missing Python packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required Python packages found")
    return True

def check_docker():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ Docker is available")
            return True
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    logger.warning("⚠️ Docker not available - database features will be limited")
    return False

def start_database(skip_db=False):
    """Start PostgreSQL database via Docker."""
    if skip_db:
        logger.info("⏭️ Skipping database startup")
        return True
    
    if not check_docker():
        logger.warning("Database startup skipped - Docker not available")
        return False
    
    logger.info("🗄️ Starting PostgreSQL database...")
    
    try:
        # Start database containers
        result = subprocess.run([
            'docker', 'compose', 'up', '-d', 'postgres'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Database started successfully")
            # Wait a bit for database to be ready
            time.sleep(5)
            return True
        else:
            logger.error(f"Failed to start database: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.error(f"Error starting database: {e}")
        return False

def initialize_database():
    """Initialize database schema."""
    logger.info("🏗️ Initializing database schema...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        from models.database_models import DatabaseManager
        
        # Initialize database
        db_manager = DatabaseManager('postgresql://postgres:yourpassword@localhost:5432/deforestation_db')
        logger.info("✅ Database schema initialized")
        return True
        
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.info("System will run in demo mode without database")
        return False

def start_web_server():
    """Start the Flask web server."""
    logger.info("🌐 Starting web server...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        from app import app
        
        # Start browser in background
        def open_browser():
            time.sleep(3)
            logger.info("🌐 Opening web browser...")
            webbrowser.open('http://localhost:5000/dashboard')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Flask server
        logger.info("✅ Web server starting on http://localhost:5000")
        logger.info("   Dashboard: http://localhost:5000/dashboard")
        logger.info("   Simple Interface: http://localhost:5000/simple")
        logger.info("   API Root: http://localhost:5000/")
        print("\n🛑 Press Ctrl+C to stop the server\n")
        print("=" * 50)
        
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.info("\n👋 System stopped by user")
    except Exception as e:
        logger.error(f"Error starting web server: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='Launch Deforestation Risk Analysis System')
    parser.add_argument('--skip-data', action='store_true', 
                       help='Skip data fetching phase')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip ML model training phase')
    parser.add_argument('--skip-db', action='store_true',
                       help='Skip database startup (demo mode)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick start - skip data, training, and use demo mode')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Quick mode shortcuts
    if args.quick:
        args.skip_data = True
        args.skip_training = True
        args.skip_db = True
        logger.info("🚀 Quick demo mode - skipping data processing and database")
    
    # Check system requirements
    logger.info("🔍 Checking system requirements...")
    if not check_python_dependencies():
        logger.error("❌ Missing required dependencies")
        sys.exit(1)
    
    # Start database (optional)
    db_available = False
    if not args.skip_db:
        db_available = start_database()
        if db_available:
            initialize_database()
    
    # Skip data fetching and ML training in this simplified launcher
    if not args.skip_data:
        logger.info("⏭️ Data fetching skipped (use separate data scripts if needed)")
    
    if not args.skip_training:
        logger.info("⏭️ ML training skipped (models will use demo predictions)")
    
    # Start web server
    logger.info("🚀 Launching web application...")
    start_web_server()

if __name__ == "__main__":
    main()