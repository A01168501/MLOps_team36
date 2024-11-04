import pandas as pd
import sys
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

def evaluate_model(model_path, X_test_path, y_test_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    y_pre = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pre,squared=False)
    r2 = r2_score(y_test, y_pre)

    return rmse, r2

if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]

    mlflow.set_tracking_uri("HTTP://localhost:5000")
    
    # Start an MLflow run
    with mlflow.start_run():
        # Evaluate the model
        rmse, r2 = evaluate_model(model_path, X_test_path, y_test_path)

        # Log parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("X_test_path", X_test_path)
        mlflow.log_param("y_test_path", y_test_path)
        
        # Log metrics in MLflow
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        
        print(f'RMSE: {rmse:.4f}')
        print(f'R²: {r2:.4f}')
        print("Evaluation metrics logged in MLflow")