"""
Configuration file for the 5G Energy Consumption Modelling Dashboard
"""

# Kaggle settings
KAGGLE_NOTEBOOK_PATH = "USERNAME/5g-energy-consumption-modelling"  # Replace with your actual username/notebook-path
NOTEBOOK_FILENAME = "models.ipynb"

# Model configurations
AVAILABLE_MODELS = [
    {
        "name": "FastAI + Keras",
        "description": "FastAI neural network combined with Keras for energy consumption prediction",
        "file": "Models/FastAI+Keras.py"
    },
    {
        "name": "TabNet + XGBoost",
        "description": "TabNet deep learning model with XGBoost for energy consumption prediction",
        "file": "Models/TabNet+XGBoost.py"
    },
    {
        "name": "Keras + XGBoost",
        "description": "Keras neural network with XGBoost for energy consumption prediction",
        "file": "Models/Keras+XGBoost.py"
    },
    {
        "name": "TabTransformer",
        "description": "TabTransformer model for energy consumption prediction",
        "file": "Models/TabTransformer.py"
    },
    # Uncommented and renamed to match the model_metrics_map
    {
        "name": "Farzi Scientist Model",
        "description": "Implementation from FarziScientist team using FastAI and Keras",
        "file": "project-FarziScientist/ecm-zindi-kp-v4-training-and-prediction-notebook.ipynb"
    },
    # Added new Random Forest model
    {
        "name": "Random Forest",
        "description": "Random Forest model with optimal hyperparameters for energy consumption prediction",
        "file": "Models/RandomForest.py"
    },
    # {
    #     "name": "TeamCake Solution", 
    #     "description": "Implementation from TeamCake using custom models",
    #     "notebook": "project-TeamCake/TestBS_in_TrainingSet.ipynb"
    # }
]

# Visualization settings
CHART_HEIGHT = 400
CHART_WIDTH = 800 