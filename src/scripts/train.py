import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

degrees = [2, 3, 4]
lr_models = []

for degree in degrees:
    model = Pipeline([
        ('polynomial', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regression', LinearRegression())
    ])
    lr_models.append((f'Linear_Regression_degree_{degree}', model))

rf_model = Pipeline([
            ('RF_regressor',RandomForestRegressor(n_estimators=100, criterion= 'squared_error', max_depth=None, max_features=10, ccp_alpha=0.01))
        ])

def train_model(X_train_path, y_train_path, model):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    model.fit(X_train, y_train.values.ravel())
    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_base_path = sys.argv[3]


    mlflow.set_tracking_uri("HTTP://localhost:5000")

    for model_name, model in lr_models:
        polynomial_degree = int(model_name.split('_')[-1])
        
        with mlflow.start_run(run_name=f"Training Linear Regression degree {polynomial_degree}"):
            trained_model = train_model(X_train_path, y_train_path, model)

            model_path = f"{model_base_path}{model_name}.pkl"
            joblib.dump(trained_model, model_path)
            mlflow.sklearn.log_model(trained_model, "model")
            mlflow.log_param("polynomial_degree", polynomial_degree)
            print(f"Model {model_name} logged in MLflow")

    with mlflow.start_run(run_name = "Training Random Forest n_estimators 100"):
        # train model
        model = train_model(X_train_path, y_train_path, rf_model)

        model_path = f"{model_base_path}Random_Forest_n_100.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("n_estimators", 100)
        print("Model Random Forest logged in MLflow")