import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mlflow

class DataPreprocessor:
    def __init__(self, data_path, target_column="CO(GT)", test_size=0.2, random_state=27):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.drop(columns=["Date", "Time", "NMHC(GT)", "Unnamed: 15", "Unnamed: 16"])
        
    def handle_missing_values(self):
        self.data = self.data.applymap(lambda x: np.nan if x == -200 else x)
        self.data = self.data.dropna(subset=[self.target_column])
        self.data = self.data.apply(lambda col: col.fillna(col.mean()))
    
    def split_data(self):
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def fit_scaler(self):
        self.scaler.fit(self.X_train)

    def transform_data(self, data):
        data_scaled = self.scaler.transform(data)
        return pd.DataFrame(data_scaled, columns=data.columns)

    def preprocess(self):
        self.load_data()
        self.handle_missing_values()
        self.split_data()
        self.fit_scaler()
        
        self.X_train = self.transform_data(self.X_train)
        self.X_test = self.transform_data(self.X_test)

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    mlflow.set_tracking_uri("HTTP://localhost:5000")

    with mlflow.start_run():
        DP = DataPreprocessor(data_path)
        DP.preprocess()
        X_train = DP.X_train 
        X_test = DP.X_test
        y_train = DP.y_train
        y_test = DP.y_test
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
