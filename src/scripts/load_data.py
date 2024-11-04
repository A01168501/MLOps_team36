import pandas as pd
import sys
import mlflow
import os

def load_data(filepath):
    return pd.read_csv(filepath)

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_file = sys.argv[2]

    mlflow.set_tracking_uri("HTTP://localhost:5000")

    with mlflow.start_run():
        data = load_data(data_path)
        data.to_csv(output_file, index=False)

        # Log the output file as an artifact
        mlflow.log_artifact(output_file)
        
        # Log parameters
        mlflow.log_param("input_file", data_path)
        mlflow.log_param("output_file", output_file)

        print("Data loaded and logged in MLflow")