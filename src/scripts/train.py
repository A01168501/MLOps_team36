import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn

polynomial_degree = 2

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    model = Pipeline([
        ('polynomial',PolynomialFeatures(degree=polynomial_degree,include_bias = False)),
        ('regression',LinearRegression())
    ])
    model.fit(X_train, y_train.values.ravel())
    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    with mlflow.start_run():
        # train model
        model = train_model(X_train_path, y_train_path)

        # save model
        joblib.dump(model, model_path)

        # log into mlflow
        mlflow.sklearn.log_model(model, "model")

        # Log parameters (in this case, the degree of the polynomial)
        mlflow.log_param("polynomial_degree", polynomial_degree)
        
        # Log metrics (you can add relevant metrics here)
        # For example, if you had a validation set, you could compute and log the error
        # mlflow.log_metric("mse", mean_squared_error(y_true, y_pred))

        print("Model logged in MLflow")