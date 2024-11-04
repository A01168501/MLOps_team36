import pytest
from scripts.preprocess_data import DataPreprocessor

@pytest.fixture
def data_preprocessor(request):
    data_path = request.config.getoption("--csv-file")
    dp = DataPreprocessor(data_path)
    dp.load_data()
    dp.handle_missing_values()
    dp.split_data()
    dp.fit_scaler()

    return dp

def test_data_no_missing_values(data_preprocessor):
    """
    Tests that there are no missing values in the training and testing datasets.

    This function asserts that both X_train and X_test contain no NaN values 
    after the preprocessing steps have been completed.

    Raises:
        AssertionError: If any of the assertions fail, indicating that 
        missing values still exist in the datasets.
    """
    
    assert not data_preprocessor.X_train.isnull().values.any(), "X_train contains missing values"
    assert not data_preprocessor.X_test.isnull().values.any(), "X_test contains missing values"

def test_scaler_preprocessing_brings_x_train_mean_near_zero(data_preprocessor):
    """
    Tests that the mean of the scaled training data
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

    original_mean = data_preprocessor.X_train.mean()
    
    data_preprocessor.X_train = data_preprocessor.transform_data(data_preprocessor.X_train)

    assert all(original_mean > data_preprocessor.X_train.mean()), "Original mean should be greater than scaled mean"
    
    scaled_mean = data_preprocessor.X_train.mean()
    assert all(0 <= scaled_mean) and all(scaled_mean <= 1), f"Mean of scaled data is {scaled_mean}"

    print(f'The mean of the original X train is: {original_mean}')
    print(f'The mean of the transformed X train is: {scaled_mean}')

def test_scaler_preprocessing_brings_x_train_std_below_one(data_preprocessor):
    """
    Tests that the standard deviation of the scaled training data 
    is below one after applying MinMaxScaler.

    This function asserts that the standard deviation of the scaled 
    training data is less than one, indicating that the scaling process 
    has effectively reduced the spread of the data.

    It also prints the standard deviation of the scaled data for reference.

    Raises:
        AssertionError: If the assertion fails, indicating that the 
        standard deviation of the scaled data is not below one.
    """
    
    data_preprocessor.X_train = data_preprocessor.transform_data(data_preprocessor.X_train)
    assert (data_preprocessor.X_train.std() < 1.0).all(), f"Standard deviation of scaled data is {data_preprocessor.X_train.std()}"
    print(f'The SD of the transformed X train is: {data_preprocessor.X_train.std()}')