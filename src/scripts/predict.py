import pandas as pd
import mlflow
import mlflow.sklearn
import sys
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import yaml
import os


def load_model(run_id):
    """
    Load the Random Forest model from MLflow using run_id
    """
    try:
        # Set the same tracking URI as in training
        mlflow.set_tracking_uri("http://localhost:5000")

        # Load the model from the specified path
        loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        return loaded_model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise



def predict_and_evaluate(X_test_path, y_test_path=None, run_id=None):
    """
    Make predictions and optionally evaluate if y_test is provided
    
    Parameters:
    - X_test_path: path to test features CSV
    - y_test_path: path to test target CSV (optional)
    - run_id: MLflow run ID for the Random Forest model
    
    Returns:
    - predictions: numpy array of predictions
    - metrics: dictionary of metrics (if y_test provided)
    """
    # Load test data
    X_test = pd.read_csv(X_test_path)
    
    # Load model
    model = load_model(run_id)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # If y_test is provided, calculate metrics
    metrics = None
    if y_test_path:
        y_test = pd.read_csv(y_test_path)
        y_test = y_test.values.ravel()
        
        metrics = {
            'r2': r2_score(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': mean_squared_error(y_test, predictions, squared=False)
        }
        
        # Log metrics to MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                'test_r2': metrics['r2'],
                'test_mse': metrics['mse'],
                'test_rmse': metrics['rmse']
            })
    
    return predictions, metrics


def load_params(params_path='params.yaml'):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

if __name__ == "__main__":
    """
    Usage:
    1. For predictions only:
       python predict.py <run_id>
    
    2. For predictions and evaluation:
       python predict.py <run_id> -e
    """
    if len(sys.argv) < 2:
        print("Usage: python predict.py <run_id>")
        sys.exit(1)
    
    # Load the parameters
    params = load_params()

    X_test_path = params['data']['processed'] + 'X_test.csv'
    run_id = sys.argv[1]
    output_path = f"results/predictions/{run_id}"
    y_test_path = params['data']['processed'] + 'Y_test.csv' if len(sys.argv) > 2 else None
    
    print(f"Running predictions on {X_test_path} with run_id {run_id}, saving results to results/predictions/{run_id}")
    if(y_test_path is not None):
        print(f"Evaluate with adata at {y_test_path}")

    # Make predictions and optionally evaluate
    predictions, metrics = predict_and_evaluate(X_test_path, y_test_path, run_id)
    
    # Save predictions
    pd.DataFrame(predictions, columns=['prediction']).to_csv(output_path, index=False)
    
    # Print metrics if available
    if metrics:
        print("\nModel Performance on Test Data:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")