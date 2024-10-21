import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Load the Air Quality dataset
data = pd.read_csv('/app/data/raw/AirQualityUCI.csv')

# Prepare data
X = data.drop(columns=['quality'])  # Features
y = data['quality']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLFlow experiment
mlflow.set_experiment("Air_Quality_Experiment")

with mlflow.start_run():
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics to MLFlow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 100)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model accuracy: {accuracy}")