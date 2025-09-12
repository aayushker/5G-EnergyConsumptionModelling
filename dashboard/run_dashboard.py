#!/usr/bin/env python3
"""
Main entry point for running the 5G Energy Consumption Modelling Dashboard
"""
import os
import sys
import streamlit.web.cli as stcli
from pathlib import Path

def run_streamlit():
    """Run the Streamlit app"""
    # Get the directory where this script is located
    dashboard_dir = Path(__file__).parent.absolute()
    
    # Set the current working directory to the dashboard directory
    os.chdir(dashboard_dir)
    
    # Construct the path to app.py
    app_path = os.path.join(dashboard_dir, "app.py")
    
    # Check if app.py exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Set up Streamlit command-line arguments
    sys.argv = [
        "streamlit", 
        "run", 
        app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",  # Allows access from other machines
        "--browser.serverAddress=localhost"
    ]
    
    # Run Streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_streamlit() 