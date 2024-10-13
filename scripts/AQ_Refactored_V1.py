import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Loading and exploring the data
def load_data(filepath): 
    data = pd.read_csv(filepath)
    return data

'''
This function will retrieve the most important features of our dataset, with this we will have a starting point of understanding how our data look like 
'''
def explore_data (data):
    print(data.head().T)
    print(data.describe())
    print(data.info())

# Exploration Data Analysis (EDA)
'''
This function will push back Nan values to the dataset, so we could further input values
'''
def input_back(data):
    data = data.applymap(lambda x: np.nan if x == -200 else x)
    return data

'''
This function will print the percentage of missing values per category or column that are present in the dataset
'''
def percentage_of_missing_values(data):
    per_miss_val = (data.isna().sum()/len(data))*100
    format_value = per_miss_val.apply(lambda x: f'{x:.2f}%')
    print("Percentage  of missing values")
    print(format_value)

# Add visualizations to the dataset
def plot_histograms(data):
    data.hist(bins=10,figsize=(10,10))
    plt.show()

# Correlation
'''
This function will plot the correlation matrix that will give us more information about the relation between the variables as well as to identify multicolineality
'''
def plot_corr_matrix(data,columns):
    data = data[columns]
    plt.figure(figsize=(10,5))
    sns.heatmap(data.corr(),annot=True, fmt=".2f",cmap= 'RdBu' )
    plt.show()

# Preprocessing and Feature Engineering
'''
This function imputes the mean in the NaN values over the samples
'''
def imputation_mean(data):
    data = data.apply(lambda col: col.fillna(col.mean()))
    return data

'''
This function will transform the independent variables of the model with a min max scaler, to fed the model scaled inputs that works better with machine learning models
'''
def scale_features(data, target):
    scaler = MinMaxScaler()
    features = data.drop(target, axis=1)
    features_scaled = scaler.fit_transform(features)
    data_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    data_scaled[target] = data[target].values
    return data_scaled

# Splitting the dataset
def split_data(data, target, test_size=0.2, random_state=27):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Training the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluating the model
def evaluate_model(model, X_text, y_test):
    y_pre = model.predict(X_text)
    rmse = mean_squared_error(y_test, y_pre,squared=False)
    r2 = r2_score(y_test, y_pre)

    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')

# Main function for running the pipeline
def main(filepath):
    data = load_data(filepath)
    explore_data(data)
    data = input_back(data)
    data = data.dropna(subset=["CO(GT)"])
    percentage_of_missing_values(data)
    data = data.drop(columns=["Date", "Time", 'NMHC(GT)', 'Unnamed: 15','Unnamed: 16'])
    
    plot_histograms(data)
    plot_corr_matrix(data, data.columns)
    
    data = imputation_mean(data)
    data_scaled = scale_features(data, 'CO(GT)')
    
    X_train, X_test, y_train, y_test = split_data(data_scaled, "CO(GT)")
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    lr_model2 = Pipeline([
        ('polynomial',PolynomialFeatures(degree=2,include_bias = False)),
        ('regression',LinearRegression())
    ])
    lr_model2.fit(X_train,y_train)
    evaluate_model(lr_model2, X_train,y_train)

if __name__ == '__main__':
    main('/home/alt9193/Documents/MLOps_team36/data/AirQualityUCI.csv')
