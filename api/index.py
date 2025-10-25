"""
Vercel entry point for Forest Analysis System
This file serves as the main entry point for Vercel deployment
"""

import sys
import os

# Add the parent directory to the Python path to import from the main app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app from the main application
from app import app

# This is the entry point for Vercel
# The app variable is what Vercel will use to serve the Flask application
handler = app
