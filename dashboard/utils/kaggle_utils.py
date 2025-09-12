import os
import json
import time
import tempfile
import requests
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleNotebookRunner:
    def __init__(self, auto_auth: bool = False):
        """Initialize the Kaggle API client
        
        Args:
            auto_auth: Whether to authenticate automatically during initialization
        """
        self.api = KaggleApi()
        self._is_authenticated = False
        if auto_auth:
            self.authenticate()
        
    def authenticate(self) -> bool:
        """
        Authenticate with Kaggle API
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            self.api.authenticate()
            self._is_authenticated = True
            return True
        except Exception as e:
            print(f"Kaggle authentication failed: {str(e)}")
            print("\nTo set up Kaggle API credentials:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token' to download kaggle.json")
            print("3. Create the directory: mkdir -p ~/.kaggle")
            print("4. Move the downloaded file: mv kaggle.json ~/.kaggle/")
            print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def ensure_authenticated(self):
        """Ensure the client is authenticated before making API calls"""
        if not self._is_authenticated and not self.authenticate():
            raise Exception("Kaggle API authentication required. Please set up your credentials.")
    
    def run_notebook(self, notebook_path: str, parameters: Dict[str, Any] = None) -> str:
        """
        Run a Kaggle notebook with optional parameters
        
        Args:
            notebook_path: Path to the notebook in the format 'username/notebook-slug'
            parameters: Dictionary of parameters to pass to the notebook
            
        Returns:
            str: ID of the notebook run
        """
        self.ensure_authenticated()
        try:
            # Parse username and notebook slug
            username, notebook_slug = notebook_path.split('/')
            
            # Create a temporary file to store the parameters
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(parameters or {}, f)
                param_file = f.name
            
            # Push the parameters to Kaggle dataset (in a real implementation)
            # For simplicity, we're just mocking this step
            
            # Create a new notebook version with the parameters
            # In production, this would use the actual Kaggle API
            # Here we're returning a mock run ID based on timestamp
            run_id = f"run_{int(time.time())}"
            
            # Clean up the temporary file
            os.unlink(param_file)
            
            return run_id
            
        except Exception as e:
            raise Exception(f"Failed to run notebook: {str(e)}")
    
    def check_run_status(self, run_id: str) -> Dict[str, Any]:
        """
        Check the status of a notebook run
        
        Args:
            run_id: ID of the notebook run
            
        Returns:
            Dict containing status information (status, progress, etc.)
        """
        try:
            # In a real implementation, this would query the Kaggle API
            # For demo purposes, we'll simulate a running notebook
            
            # Parse the timestamp from the run_id to simulate realistic progress
            timestamp = int(run_id.split('_')[1])
            elapsed_seconds = time.time() - timestamp
            
            # Simulate different states based on elapsed time
            if elapsed_seconds < 5:
                status = "queued"
                progress = 0.0
            elif elapsed_seconds < 60:
                status = "running"
                progress = min(elapsed_seconds / 180, 0.99)  # Max out at 99%
            else:
                status = "complete"
                progress = 1.0
                
            return {
                "status": status,
                "progress": progress,
                "elapsed_time": elapsed_seconds
            }
            
        except Exception as e:
            raise Exception(f"Failed to check run status: {str(e)}")
    
    def get_run_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed notebook run
        
        Args:
            run_id: ID of the notebook run
            
        Returns:
            Dict containing the results (metrics, output files, etc.)
        """
        try:
            # Check if the run is complete
            status = self.check_run_status(run_id)
            if status["status"] != "complete":
                raise Exception("Notebook run is not complete yet")
            
            # In a real implementation, this would download the output from Kaggle
            # For demo purposes, we'll return mock results
            
            # Generate mock metrics
            metrics = {
                "mae": 0.035 + (hash(run_id) % 100) / 1000,
                "mse": 0.0023 + (hash(run_id) % 100) / 10000,
                "wmape": 0.054 + (hash(run_id) % 100) / 1000,
                "training_time": 142.3 + (hash(run_id) % 100)
            }
            
            # Generate mock prediction data
            # In a real implementation, this would be downloaded from Kaggle
            
            return {
                "metrics": metrics,
                "output_files": ["predictions.csv", "model.pkl"],
                "logs": "Training completed successfully"
            }
            
        except Exception as e:
            raise Exception(f"Failed to get run results: {str(e)}")
    
    def stop_run(self, run_id: str) -> bool:
        """
        Stop a running notebook
        
        Args:
            run_id: ID of the notebook run
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            # In a real implementation, this would call the Kaggle API to stop the run
            # For demo purposes, we'll just return True
            return True
            
        except Exception as e:
            raise Exception(f"Failed to stop run: {str(e)}")
    
    def list_available_notebooks(self, username: str) -> List[Dict[str, Any]]:
        """
        List available notebooks for a user
        
        Args:
            username: Kaggle username
            
        Returns:
            List of dictionaries with notebook information
        """
        try:
            # In a real implementation, this would call the Kaggle API
            # api.kernels_list(username=username)
            
            # For demo purposes, return mock data
            return [
                {
                    "title": "5G Energy Consumption Model",
                    "slug": "5g-energy-consumption-model",
                    "url": f"https://www.kaggle.com/{username}/5g-energy-consumption-model",
                    "lastRunTime": "2023-06-15T12:34:56Z"
                },
                {
                    "title": "Energy Prediction Advanced",
                    "slug": "energy-prediction-advanced",
                    "url": f"https://www.kaggle.com/{username}/energy-prediction-advanced",
                    "lastRunTime": "2023-06-10T15:45:12Z"
                }
            ]
            
        except Exception as e:
            raise Exception(f"Failed to list notebooks: {str(e)}")
    
    def download_notebook_output(self, run_id: str, output_path: str) -> List[str]:
        """
        Download output files from a notebook run
        
        Args:
            run_id: ID of the notebook run
            output_path: Directory to save output files
            
        Returns:
            List of paths to downloaded files
        """
        try:
            # In a real implementation, this would download files using the Kaggle API
            # For demo purposes, we'll just return mock file paths
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Return mock file paths
            return [
                os.path.join(output_path, "predictions.csv"),
                os.path.join(output_path, "model.pkl")
            ]
            
        except Exception as e:
            raise Exception(f"Failed to download output: {str(e)}")
    
    def get_notebook_parameters(self, notebook_path: str) -> Dict[str, Any]:
        """
        Get the parameters of a notebook
        
        Args:
            notebook_path: Path to the notebook in the format 'username/notebook-slug'
            
        Returns:
            Dict of notebook parameters
        """
        try:
            # In a real implementation, this would parse the notebook JSON
            # For demo purposes, return mock parameters
            return {
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.001,
                "patience": 5,
                "validation_split": 0.2
            }
            
        except Exception as e:
            raise Exception(f"Failed to get notebook parameters: {str(e)}")

def format_run_time(seconds: float) -> str:
    """Format run time in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours" 