import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataPreparation:
    def __init__(self, data_dir='MLmodel/Dataset'):
        """
        Initialize the DataPreparation class.
        
        Parameters:
        data_dir (str): Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.bs_config = None
        self.time_records = None
        self.load_data = None
        self.energy_data = None
        self.merged_data = None
        
    def load_datasets(self):
        """
        Load all datasets from the data directory.
        """
        print("Loading datasets...")
        
        # Load base station configuration data
        self.bs_config = pd.read_csv(os.path.join(self.data_dir, 'imgs_2023071012123392536.csv'))
        print(f"Base station config loaded: {self.bs_config.shape}")
        
        # Load timestamped records
        self.time_records = pd.read_csv(os.path.join(self.data_dir, 'imgs_202307101549519358.csv'))
        print(f"Time records loaded: {self.time_records.shape}")
        
        # Load load and energy saving modes data
        self.load_data = pd.read_csv(os.path.join(self.data_dir, 'imgs_2023071012130978799.csv'))
        print(f"Load data loaded: {self.load_data.shape}")
        
        # Load energy consumption data
        self.energy_data = pd.read_csv(os.path.join(self.data_dir, 'imgs_2023071012133740345.csv'))
        print(f"Energy data loaded: {self.energy_data.shape}")
        
        return True
        
    def standardize_timestamps(self):
        """
        Standardize timestamp formats across all datasets.
        """
        print("Standardizing timestamps...")
        
        # For load data - format: M/D/YYYY H:MM
        self.load_data['Time'] = pd.to_datetime(self.load_data['Time'], format='%m/%d/%Y %H:%M', errors='coerce')
        
        # For energy data - format: M/D/YYYY H:MM
        self.energy_data['Time'] = pd.to_datetime(self.energy_data['Time'], format='%m/%d/%Y %H:%M', errors='coerce')
        
        # For time records - format: YYYY-MM-DD HH:MM:SS
        self.time_records['Time'] = pd.to_datetime(self.time_records['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Drop rows with invalid timestamps
        self.load_data = self.load_data.dropna(subset=['Time'])
        self.energy_data = self.energy_data.dropna(subset=['Time'])
        self.time_records = self.time_records.dropna(subset=['Time'])
        
        # Round timestamps to the nearest hour to ensure alignment
        self.load_data['Time'] = self.load_data['Time'].dt.floor('H')
        self.energy_data['Time'] = self.energy_data['Time'].dt.floor('H')
        self.time_records['Time'] = self.time_records['Time'].dt.floor('H')
        
        print(f"After timestamp standardization - Load data: {self.load_data.shape}, Energy data: {self.energy_data.shape}")
        return True
        
    def merge_datasets(self):
        """
        Merge the datasets to create a unified dataset for modeling.
        """
        print("Merging datasets...")
        
        # Start with the load data as the base
        merged = self.load_data.copy()
        
        # Merge with energy data
        merged = pd.merge(
            merged, 
            self.energy_data[['Time', 'BS', 'Energy']], 
            on=['Time', 'BS'], 
            how='left'
        )
        
        # Merge with base station configuration
        merged = pd.merge(
            merged,
            self.bs_config,
            on='BS',
            how='left'
        )
        
        self.merged_data = merged
        print(f"Merged data shape: {self.merged_data.shape}")
        
        # Check for missing values
        missing_values = self.merged_data.isna().sum()
        print("Missing values in merged data:")
        print(missing_values[missing_values > 0])
        
        return True
        
    def handle_missing_values(self):
        """
        Handle missing values in the merged dataset.
        """
        print("Handling missing values...")
        
        # Check for missing energy values
        missing_energy = self.merged_data['Energy'].isna().sum()
        print(f"Missing Energy values: {missing_energy} ({missing_energy/len(self.merged_data)*100:.2f}%)")
        
        if missing_energy > 0:
            # For Energy, interpolate missing values by BS
            self.merged_data = self.merged_data.sort_values(['BS', 'Time'])
            self.merged_data['Energy'] = self.merged_data.groupby('BS')['Energy'].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        
        # For other columns, fill with mode or median as appropriate
        for col in self.merged_data.columns:
            if col not in ['Time', 'BS', 'Energy']:
                missing = self.merged_data[col].isna().sum()
                if missing > 0:
                    if self.merged_data[col].dtype == 'object':
                        # For categorical columns, fill with mode
                        mode_val = self.merged_data[col].mode()[0]
                        self.merged_data[col] = self.merged_data[col].fillna(mode_val)
                    else:
                        # For numerical columns, fill with median
                        median_val = self.merged_data[col].median()
                        self.merged_data[col] = self.merged_data[col].fillna(median_val)
        
        # Check remaining missing values
        missing_after = self.merged_data.isna().sum()
        print("Missing values after handling:")
        print(missing_after[missing_after > 0])
        
        return True
    
    def add_time_features(self):
        """
        Add time-based features to the dataset.
        """
        print("Adding time features...")
        
        # Extract time-based features
        self.merged_data['hour'] = self.merged_data['Time'].dt.hour
        self.merged_data['day_of_week'] = self.merged_data['Time'].dt.dayofweek
        self.merged_data['is_weekend'] = self.merged_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical time features (hour of day) to capture cyclical nature of time
        self.merged_data['hour_sin'] = np.sin(2 * np.pi * self.merged_data['hour'] / 24)
        self.merged_data['hour_cos'] = np.cos(2 * np.pi * self.merged_data['hour'] / 24)
        
        print(f"Time features added. Data shape: {self.merged_data.shape}")
        return True
    
    def save_prepared_data(self, output_file='ClaudeProject/prepared_data.csv'):
        """
        Save the prepared dataset to a CSV file.
        
        Parameters:
        output_file (str): Path to save the prepared data
        """
        print(f"Saving prepared data to {output_file}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        self.merged_data.to_csv(output_file, index=False)
        
        print(f"Data saved successfully. Shape: {self.merged_data.shape}")
        return True
    
    def prepare_data(self, output_file='ClaudeProject/prepared_data.csv'):
        """
        Execute the full data preparation pipeline.
        
        Parameters:
        output_file (str): Path to save the prepared data
        
        Returns:
        pd.DataFrame: Prepared data
        """
        self.load_datasets()
        self.standardize_timestamps()
        self.merge_datasets()
        self.handle_missing_values()
        self.add_time_features()
        self.save_prepared_data(output_file)
        
        return self.merged_data

if __name__ == "__main__":
    # Execute the data preparation pipeline
    prep = DataPreparation()
    prepared_data = prep.prepare_data()
    
    # Display summary of the prepared data
    print("\nPrepared Data Summary:")
    print(f"Shape: {prepared_data.shape}")
    print("\nSample of prepared data:")
    print(prepared_data.head())
    
    print("\nColumn information:")
    print(prepared_data.info())
    
    print("\nBasic statistics:")
    print(prepared_data.describe()) 