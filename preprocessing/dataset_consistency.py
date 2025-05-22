import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

def validate_model_performance():
    """
    Run a series of tests to validate if the model is truly generalizing
    or if there's data leakage/overfitting.
    """
    # Load the engineered data
    print("Loading engineered data...")
    data = pd.read_csv('engineered_data.csv')

    # Convert Time to datetime
    if 'Time' in data.columns:
        data['Time'] = pd.to_datetime(data['Time'])

    # Sort by time
    data = data.sort_values('Time')

    # Define target and exclude non-feature columns
    target_col = 'Energy'
    exclude_cols = ['Time', 'BS', 'CellName', target_col]

    # Create different feature sets to test
    print("\n1. Testing different feature sets...")

    # All features
    all_features = [col for col in data.columns if col not in exclude_cols]

    # No energy-derived features
    energy_features = [col for col in all_features if 'energy' in col.lower()]
    non_energy_features = [col for col in all_features if col not in energy_features]

    print(f"Total features: {len(all_features)}")
    print(f"Energy-derived features: {len(energy_features)}")
    print(f"Non-energy features: {len(non_energy_features)}")

    feature_sets = {
        'all_features': all_features,
        'no_energy_features': non_energy_features
    }

    # Run tests with different temporal splits
    results = {}

    # Test 1: Simple random forest with different feature sets
    print("\n2. Training models with different feature sets...")

    for name, features in feature_sets.items():
        # Create X and y
        X = data[features]
        y = data[target_col]

        # Split data - 80% train, 20% test
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train a simple random forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)

        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

        print(f"{name}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")

    # Test 2: Forward-chaining cross validation (more realistic for time series)
    print("\n3. Testing with time-series cross-validation...")

    # Create 5 time-based folds
    tscv = TimeSeriesSplit(n_splits=5)

    for name, features in feature_sets.items():
        cv_results = []

        X = data[features]
        y = data[target_col]

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            test_preds = model.predict(X_test)
            test_r2 = r2_score(y_test, test_preds)
            cv_results.append(test_r2)

        print(f"{name} time-series CV R²: {np.mean(cv_results):.4f} (std: {np.std(cv_results):.4f})")

    # Test 3: Test on distant future data
    print("\n4. Testing on distant future data...")

    # Get the earliest and latest dates
    earliest_date = data['Time'].min()
    latest_date = data['Time'].max()
    range_days = (latest_date - earliest_date).days

    # Define train and test periods with a significant gap
    # Train on first 60%, test on last 20%
    train_end_date = earliest_date + pd.Timedelta(days=int(range_days * 0.6))
    test_start_date = latest_date - pd.Timedelta(days=int(range_days * 0.2))

    print(f"Train period: {earliest_date} to {train_end_date}")
    print(f"Test period: {test_start_date} to {latest_date}")

    for name, features in feature_sets.items():
        # Create train and test sets
        train_mask = data['Time'] <= train_end_date
        test_mask = data['Time'] >= test_start_date

        X_train = data.loc[train_mask, features]
        y_train = data.loc[train_mask, target_col]
        X_test = data.loc[test_mask, features]
        y_test = data.loc[test_mask, target_col]

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)

        print(f"{name} distant future R²: Train={train_r2:.4f}, Test={test_r2:.4f}")

    # Test 4: Feature importance
    print("\n5. Top 20 important features:")

    X = data[all_features]
    y = data[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print top 20 features
    top_features = [(all_features[i], importances[i]) for i in indices[:20]]
    for i, (feature, importance) in enumerate(top_features):
        print(f"{i+1}. {feature}: {importance:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.barh(range(20), [imp for _, imp in top_features][::-1])
    plt.yticks(range(20), [feat for feat, _ in top_features][::-1])
    plt.tight_layout()
    plt.savefig('feature_importance_validation.png')

    print("\nFeature importance plot saved to 'feature_importance_validation.png'")

    return results

if __name__ == "__main__":
    results = validate_model_performance()
    print("\nValidation complete.")