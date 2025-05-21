import os
import time
import argparse
from preprocessing.data_preparation import DataPreparation
from preprocessing.feature_engineering import FeatureEngineering
from model.model_training import ModelTraining

def run_pipeline(tune_model=False, output_dir='output'):
    """
    Run the full energy consumption prediction pipeline from data preparation to model evaluation.
    
    Parameters:
    tune_model (bool): Whether to perform hyperparameter tuning
    output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set file paths
    prepared_data_path = os.path.join(output_dir, 'prepared_data.csv')
    engineered_data_path = os.path.join(output_dir, 'engineered_data.csv')
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Data Preparation
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    data_prep = DataPreparation(data_dir='../MLmodel/Dataset')
    data_prep.prepare_data(output_file=prepared_data_path)
    
    # Step 2: Feature Engineering
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    feature_eng = FeatureEngineering(data_file=prepared_data_path)
    feature_eng.engineer_features(
        output_file=engineered_data_path,
        correlation_plot=os.path.join(output_dir, 'feature_correlations.png')
    )
    
    # Step 3: Model Training and Evaluation
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*80)
    model_trainer = ModelTraining(data_file=engineered_data_path)
    best_model = model_trainer.train_and_evaluate(tune=tune_model)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Best model: {best_model}")
    print(f"Total runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Output files saved to: {os.path.abspath(output_dir)}")
    
    return best_model

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Energy Consumption Prediction Pipeline')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output files')
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(tune_model=args.tune, output_dir=args.output_dir) 