import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineering:
    def __init__(self, data_file='prepared_data.csv'):
        """
        Initialize the FeatureEngineering class.
        
        Parameters:
        data_file (str): Path to the prepared data file
        """
        self.data_file = data_file
        self.data = None
        self.processed_data = None
        self.feature_importance = None
        
    def load_data(self):
        """
        Load the prepared data.
        """
        print(f"Loading data from {self.data_file}...")
        self.data = pd.read_csv(self.data_file)
        print(f"Data loaded with shape: {self.data.shape}")
        return True
    
    def create_load_features(self):
        """
        Create features based on load patterns.
        """
        print("Creating load-based features...")
        
        # Sort by BS and Time for proper lag creation
        self.data = self.data.sort_values(['BS', 'Time'])
        
        # Create lag features of load (previous hours)
        for lag in [1, 2, 3, 6, 12, 24]:
            self.data[f'load_lag_{lag}'] = self.data.groupby('BS')['load'].shift(lag)
        
        # Create rolling statistics of load
        for window in [3, 6, 12, 24]:
            # Rolling average
            self.data[f'load_rolling_mean_{window}'] = self.data.groupby('BS')['load'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling maximum
            self.data[f'load_rolling_max_{window}'] = self.data.groupby('BS')['load'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Rolling minimum
            self.data[f'load_rolling_min_{window}'] = self.data.groupby('BS')['load'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            # Rolling standard deviation (volatility)
            self.data[f'load_rolling_std_{window}'] = self.data.groupby('BS')['load'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Create load change features (rate of change)
        self.data['load_change_1h'] = self.data.groupby('BS')['load'].diff()
        self.data['load_change_rate_1h'] = self.data.groupby('BS')['load'].pct_change()
        
        # Create load acceleration (change in rate of change)
        self.data['load_acceleration'] = self.data.groupby('BS')['load_change_1h'].diff()
        
        print(f"Load features created. Data shape: {self.data.shape}")
        return True
    
    def create_time_pattern_features(self):
        """
        Create additional time-based pattern features.
        """
        print("Creating time pattern features...")
        
        # Extract or convert existing time features if they don't exist
        if 'Time' in self.data.columns and pd.api.types.is_object_dtype(self.data['Time']):
            self.data['Time'] = pd.to_datetime(self.data['Time'])
        
        if 'hour' not in self.data.columns and 'Time' in self.data.columns:
            self.data['hour'] = pd.to_datetime(self.data['Time']).dt.hour
            self.data['day_of_week'] = pd.to_datetime(self.data['Time']).dt.dayofweek
            self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create peak hours flag (typical high usage periods)
        morning_peak = (self.data['hour'] >= 7) & (self.data['hour'] <= 10)
        evening_peak = (self.data['hour'] >= 17) & (self.data['hour'] <= 22)
        self.data['is_peak_hour'] = (morning_peak | evening_peak).astype(int)
        
        # Create off-peak hours flag (typically low usage periods)
        self.data['is_night'] = ((self.data['hour'] >= 23) | (self.data['hour'] <= 5)).astype(int)
        
        # Create sine and cosine transformations for day of week (cyclical feature)
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        print(f"Time pattern features created. Data shape: {self.data.shape}")
        return True
    
    def create_energy_features(self):
        """
        Create features based on energy consumption patterns.
        """
        print("Creating energy-based features...")
        
        # Energy lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            self.data[f'energy_lag_{lag}'] = self.data.groupby('BS')['Energy'].shift(lag)
        
        # Rolling statistics for energy
        for window in [3, 6, 12, 24]:
            # Rolling average
            self.data[f'energy_rolling_mean_{window}'] = self.data.groupby('BS')['Energy'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling maximum
            self.data[f'energy_rolling_max_{window}'] = self.data.groupby('BS')['Energy'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Rolling standard deviation
            self.data[f'energy_rolling_std_{window}'] = self.data.groupby('BS')['Energy'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Energy rate of change
        self.data['energy_change_1h'] = self.data.groupby('BS')['Energy'].diff()
        self.data['energy_change_rate_1h'] = self.data.groupby('BS')['Energy'].pct_change()
        
        print(f"Energy features created. Data shape: {self.data.shape}")
        return True
    
    def create_interaction_features(self):
        """
        Create interaction features between different variables.
        """
        print("Creating interaction features...")
        
        # Load and configuration interactions
        if 'Frequency' in self.data.columns:
            self.data['load_freq_interaction'] = self.data['load'] * self.data['Frequency']
        
        if 'Bandwidth' in self.data.columns:
            self.data['load_bandwidth_interaction'] = self.data['load'] * self.data['Bandwidth']
        
        if 'Antennas' in self.data.columns:
            self.data['load_antennas_interaction'] = self.data['load'] * self.data['Antennas']
        
        if 'TXpower' in self.data.columns:
            self.data['load_txpower_interaction'] = self.data['load'] * self.data['TXpower']
        
        # Time and load interactions
        self.data['hour_load_interaction'] = self.data['hour'] * self.data['load']
        self.data['weekend_load_interaction'] = self.data['is_weekend'] * self.data['load']
        
        # Create efficiency metric (if both load and energy exist)
        if 'load' in self.data.columns and 'Energy' in self.data.columns:
            # Avoid division by zero
            min_load = max(0.01, self.data['load'].min())
            self.data['energy_efficiency'] = self.data['Energy'] / self.data['load'].clip(lower=min_load)
        
        print(f"Interaction features created. Data shape: {self.data.shape}")
        return True
    
    def encode_categorical_features(self):
        """
        Encode categorical features for machine learning models.
        """
        print("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and col not in ['Time', 'BS', 'CellName']:
                categorical_cols.append(col)
        
        if categorical_cols:
            print(f"Categorical columns to encode: {categorical_cols}")
            
            # For each categorical column, create one-hot encoding
            for col in categorical_cols:
                # Get dummies and add prefix to avoid column name conflicts
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                # Concatenate the encoded columns to the dataframe
                self.data = pd.concat([self.data, dummies], axis=1)
                # Drop the original categorical column
                self.data = self.data.drop(col, axis=1)
        else:
            print("No categorical columns to encode.")
        
        print(f"Categorical encoding completed. Data shape: {self.data.shape}")
        return True
    
    def handle_missing_values(self):
        """
        Handle any missing values created during feature engineering.
        """
        print("Handling missing values after feature engineering...")
        
        # Check for missing values
        missing_values = self.data.isna().sum()
        missing_cols = missing_values[missing_values > 0]
        
        if len(missing_cols) > 0:
            print(f"Found {len(missing_cols)} columns with missing values:")
            print(missing_cols)
            
            # Fill missing values based on column type
            for col in missing_cols.index:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                
                # For lag and time-based features, forward fill within each BS group
                if 'lag' in col or 'rolling' in col or 'change' in col:
                    self.data[col] = self.data.groupby('BS')[col].transform(
                        lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    )
                else:
                    # For other numeric columns, fill with median
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
        
        # Verify missing values have been handled
        missing_after = self.data.isna().sum()
        if missing_after.sum() > 0:
            print("Remaining missing values:")
            print(missing_after[missing_after > 0])
            # For any remaining missing values, fill with 0
            self.data = self.data.fillna(0)
        else:
            print("All missing values have been handled.")
        
        return True
    
    def visualize_feature_correlations(self, output_file='feature_correlations.png'):
        """
        Visualize correlations between features and the target variable.
        
        Parameters:
        output_file (str): Path to save the correlation plot
        """
        print("Visualizing feature correlations with Energy...")
        
        # Select numeric columns only
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # If too many columns, select the top correlated ones
        if len(numeric_cols) > 30:  # Limit to top correlations for readability
            correlations = self.data[numeric_cols].corr()['Energy'].abs().sort_values(ascending=False)
            top_correlated = correlations.head(30).index.tolist()
            numeric_cols = [col for col in top_correlated if col in numeric_cols]
        
        # Create correlation plot
        plt.figure(figsize=(16, 14))
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlations Heatmap')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Correlation plot saved to {output_file}")
        return True
    
    def save_engineered_data(self, output_file='engineered_data.csv'):
        """
        Save the engineered dataset to a CSV file.
        
        Parameters:
        output_file (str): Path to save the engineered data
        """
        print(f"Saving engineered data to {output_file}...")
        
        # Remove any duplicate columns if they exist
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        
        # Save to CSV
        self.data.to_csv(output_file, index=False)
        
        print(f"Engineered data saved successfully. Shape: {self.data.shape}")
        return True
    
    def engineer_features(self, output_file='engineered_data.csv', correlation_plot='feature_correlations.png'):
        """
        Execute the full feature engineering pipeline.
        
        Parameters:
        output_file (str): Path to save the engineered data
        correlation_plot (str): Path to save the correlation plot
        
        Returns:
        pd.DataFrame: Engineered data
        """
        self.load_data()
        self.create_load_features()
        self.create_time_pattern_features()
        self.create_energy_features()
        self.create_interaction_features()
        self.encode_categorical_features()
        self.handle_missing_values()
        self.visualize_feature_correlations(correlation_plot)
        self.save_engineered_data(output_file)
        
        return self.data

if __name__ == "__main__":
    # Execute the feature engineering pipeline
    fe = FeatureEngineering()
    engineered_data = fe.engineer_features()
    
    # Display summary of the engineered data
    print("\nEngineered Data Summary:")
    print(f"Shape: {engineered_data.shape}")
    print(f"Number of features: {engineered_data.shape[1] - 1}")  # Excluding target variable
    
    print("\nSample of engineered data:")
    print(engineered_data.head())
    
    # Print the columns for reference
    print("\nAll columns in engineered dataset:")
    for i, col in enumerate(engineered_data.columns):
        print(f"{i+1}. {col}") 