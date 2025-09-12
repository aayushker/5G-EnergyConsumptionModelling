# 5G Energy Consumption Modelling Dashboard

A Streamlit-based dashboard for running and monitoring 5G energy consumption prediction models on Kaggle.

## Overview

This dashboard provides a user-friendly interface to:

- Run various energy consumption prediction models on Kaggle
- Configure model parameters
- Monitor training progress
- View and analyze results
- Compare performance across models

## Installation

1. Ensure you have the necessary dependencies installed:

```bash
pip install streamlit pandas numpy matplotlib plotly kaggle
```

2. Set up your Kaggle API credentials:
   - Go to your Kaggle account settings
   - Click "Create New API Token" to download kaggle.json
   - Place the kaggle.json file in the ~/.kaggle/ directory
   - Ensure permissions are set correctly: `chmod 600 ~/.kaggle/kaggle.json`

3. Update the configuration:
   - Edit `config.py` to set your Kaggle notebook path

## Usage

Run the dashboard from the project root directory:

```bash
python run_dashboard.py
```

Or directly from the dashboard directory:

```bash
cd dashboard
python run_dashboard.py
```

The dashboard will be available at http://localhost:8501

## Features

- **Model Selection**: Choose from multiple 5G energy consumption prediction models
- **Parameter Configuration**: Customize training parameters
- **Real-time Monitoring**: Track training progress
- **Result Visualization**: View predictions and model performance
- **Model Comparison**: Compare metrics across different models
- **Run History**: Keep track of past model runs

## Integration with Kaggle

The dashboard communicates with Kaggle through the Kaggle API to:
1. Push parameters to notebooks
2. Trigger notebook execution
3. Monitor execution status
4. Retrieve results

## Troubleshooting

- **Authentication Issues**: Ensure your Kaggle API credentials are correctly set up
- **Connection Problems**: Check your internet connection and Kaggle API status
- **Missing Dependencies**: Verify all required packages are installed
- **Port Already In Use**: If port 8501 is already in use, modify the port in `run_dashboard.py` 