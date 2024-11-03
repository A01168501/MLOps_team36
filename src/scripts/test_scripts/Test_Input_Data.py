import pandas as pd
import numpy as np
import pytest
import sys

AQ_schema = {
    "Date": {
        "range": {
            "min": None,
            "max": None
        },
        "dtype": "object"
    },
    "Time": {
        "range": {
            "min": None,
            "max": None
        },
        "dtype": "object"
    },
    "CO(GT)": {
        "range": {
            "min": -200.0,
            "max": 11.9
        },
        "dtype": "float64"
    },
    "PT08.S1(CO)": {
        "range": {
            "min": -200.0,
            "max": 2040.0
        },
        "dtype": "float64"
    },
    "NMHC(GT)": {
        "range": {
            "min": -200.0,
            "max": 1189.0
        },
        "dtype": "float64"
    },
    "C6H6(GT)": {
        "range": {
            "min": -200.0,
            "max": 63.7
        },
        "dtype": "float64"
    },
    "PT08.S2(NMHC)": {
        "range": {
            "min": -200.0,
            "max": 2214.0
        },
        "dtype": "float64"
    },
    "NOx(GT)": {
        "range": {
            "min": -200.0,
            "max": 1479.0
        },
        "dtype": "float64"
    },
    "PT08.S3(NOx)": {
        "range": {
            "min": -200.0,
            "max": 2683.0
        },
        "dtype": "float64"
    },
    "NO2(GT)": {
        "range": {
            "min": -200.0,
            "max": 340.0
        },
        "dtype": "float64"
    },
    "PT08.S4(NO2)": {
        "range": {
            "min": -200.0,
            "max": 2775.0
        },
        "dtype": "float64"
    },
    "PT08.S5(O3)": {
        "range": {
            "min": -200.0,
            "max": 2523.0
        },
        "dtype": "float64"
    },
    "T": {
        "range": {
            "min": -200.0,
            "max": 44.6
        },
        "dtype": "float64"
    },
    "RH": {
        "range": {
            "min": -200.0,
            "max": 88.7
        },
        "dtype": "float64"
    },
    "AH": {
        "range": {
            "min": -200.0,
            "max": 2.231
        },
        "dtype": "float64"
    },
    "Unnamed: 15": {
        "range": {
            "min": float('nan'),
            "max": float('nan')
        },
        "dtype": "float64"
    },
    "Unnamed: 16": {
        "range": {
            "min": float('nan'),
            "max": float('nan')
        },
        "dtype": "float64"
    }
}

frame = {}
# Obtener la ruta del archivo CSV desde los argumentos
csv_path = '/home/alt9193/Documents/MLOps_team36/data/raw/AirQualityUCI.csv'

# Cargar el archivo CSV
df = pd.read_csv(csv_path)

# Construir el diccionario 'frame' con min, max y dtype de cada columna
for column in df.columns:
    # Obtener el tipo de dato original
    dtype = str(df[column].dtype)
    
    # Establecer min y max como None si es un tipo object
    if dtype == 'object':
        min_val = max_val = None
    else:
        # Para otros tipos de datos, calcular min y max
        min_val = df[column].min()
        max_val = df[column].max()

    # Crear el diccionario para la columna actual en el esquema
    frame[column] = {
        'range': {
            'min': min_val,
            'max': max_val
        },
        'dtype': dtype
    }


def test_input_data_ranges():
    for feature in AQ_schema:
        if frame[feature]['dtype'] != 'object':
            if np.isnan(AQ_schema[feature]['range']['max']):
                assert np.isnan(frame[feature]['range']['max'])
            else:
                assert frame[feature]['range']['max'] <= AQ_schema[feature]['range']['max']
            if np.isnan(AQ_schema[feature]['range']['min']):
                assert np.isnan(frame[feature]['range']['min'])
            else:
                assert frame[feature]['range']['min'] <= AQ_schema[feature]['range']['min']

def test_input_data_types():
    # Verificar la compatibilidad entre tipos de datos
    for feature in AQ_schema:
        assert frame[feature]['dtype'] == AQ_schema[feature]['dtype']
