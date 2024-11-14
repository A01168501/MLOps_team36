def pytest_addoption(parser):
    parser.addoption("--csv-file", action="store", default="../data/raw/AirQualityUCI.csv", help="Path to the CSV file for testing")
