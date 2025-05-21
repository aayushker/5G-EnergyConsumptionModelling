#!/usr/bin/env python3
"""
Launcher for the 5G Energy Consumption Modelling Dashboard
"""
import os
import sys
from pathlib import Path

def main():
    # Get the absolute path to the dashboard directory
    dashboard_dir = Path(__file__).parent.absolute() / "dashboard"
    
    # Check if the dashboard directory exists
    if not dashboard_dir.exists():
        print(f"Error: Dashboard directory not found at {dashboard_dir}")
        sys.exit(1)
    
    # Construct the path to the run_dashboard.py script
    run_script = dashboard_dir / "run_dashboard.py"
    
    # Check if the run script exists
    if not run_script.exists():
        print(f"Error: Run script not found at {run_script}")
        sys.exit(1)
    
    # Make the script executable if it's not already
    if not os.access(run_script, os.X_OK):
        os.chmod(run_script, 0o755)
    
    # Execute the dashboard script
    print("Starting 5G Energy Consumption Modelling Dashboard...")
    os.execv(sys.executable, [sys.executable, str(run_script)])

if __name__ == "__main__":
    main() 