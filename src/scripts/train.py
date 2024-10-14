import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    model = Pipeline([
        ('polynomial',PolynomialFeatures(degree=2,include_bias = False)),
        ('regression',LinearRegression())
    ])
    model.fit(X_train, y_train.values.ravel())
    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(X_train_path, y_train_path)
    joblib.dump(model, model_path)