import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mlflow

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=["Date", "Time", "NMHC(GT)", "Unnamed: 15","Unnamed: 16"])
    data.applymap(lambda x: np.nan if x == -200 else x)
    data = data.dropna(subset=["CO(GT)"])
    data = data.apply(lambda col: col.fillna(col.mean()))
    
    X = data.drop("CO(GT)", axis=1)
    y = data["CO(GT)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = preprocess_data(data_path)
        pd.DataFrame(X_train).to_csv(output_train_features, index=False)
        pd.DataFrame(X_test).to_csv(output_test_features, index=False)
        pd.DataFrame(y_train).to_csv(output_train_target, index=False)
        pd.DataFrame(y_test).to_csv(output_test_target, index=False)

        # Log the output file paths as artifacts
        mlflow.log_artifact(output_train_features)
        mlflow.log_artifact(output_test_features)
        mlflow.log_artifact(output_train_target)
        mlflow.log_artifact(output_test_target)

         # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 27)

        print("Preprocessing completed and logged in MLflow")
