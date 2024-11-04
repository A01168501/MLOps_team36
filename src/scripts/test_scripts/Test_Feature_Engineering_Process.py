import pandas as pd
import numpy as np
import pytest
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_path = '/home/alt9193/Documents/MLOps_team36/data/raw/AirQualityUCI.csv'
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

def test_scaler_preprocessing_brings_x_train_mean_near_zero():
    """
    Tests that the mean of the scaled training data (X_train_scaled) 
    is near zero after applying MinMaxScaler.

    This function calculates the original mean of the training data (X_train) 
    and then checks the following conditions:
    
    1. Asserts that the original mean of the training data is greater than 
       the mean of the scaled data. This ensures that the scaling process 
       has effectively reduced the mean.
       
    2. Asserts that the mean of the scaled data lies between 0 and 1, 
       inclusive. This verifies that the MinMaxScaler has transformed the 
       data correctly within the expected range.

    Additionally, it prints the original and scaled means for reference.

    Raises:
        AssertionError: If any of the assertions fail, indicating that the 
        scaling process did not produce the expected results.
    """
    original_mean = X_train.stack().mean()
    
    assert original_mean > X_train_scaled.mean()
    assert 0 <= X_train_scaled.mean() <= 1, f"Mean of scaled data is {X_train_scaled.mean()}"

    print(f'The mean of the original X train is: {original_mean}')
    print(f'The mean of the transformed X train is: {X_train_scaled.mean()}')

def test_scaler_preprocessing_brings_x_train_std_below_one():
    """
    Tests that the standard deviation of the scaled training data 
    (X_train_scaled) is below one after applying MinMaxScaler.

    This function asserts that the standard deviation of the scaled 
    training data is less than one, indicating that the scaling process 
    has effectively reduced the spread of the data.

    It also prints the standard deviation of the scaled data for reference.

    Raises:
        AssertionError: If the assertion fails, indicating that the 
        standard deviation of the scaled data is not below one.
    """
    assert X_train_scaled.std() < 1.0, f"Standard deviation of scaled data is {X_train_scaled.std()}"
    print(f'The SD of the transformed X train is: {X_train_scaled.std()}')