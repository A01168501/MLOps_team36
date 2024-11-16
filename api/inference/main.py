import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import onnxruntime

# Ruta del archivo de configuración y de modelos
MODEL_JSON_PATH = "models/deploy/model.json"
MODEL_DIR = "models/onnx"

# Verificar la existencia del archivo JSON
if not os.path.exists(MODEL_JSON_PATH):
    raise FileNotFoundError(f"El archivo de configuración {MODEL_JSON_PATH} no se encontró.")

# Cargar configuración del modelo
with open(MODEL_JSON_PATH, "r") as f:
    config = json.load(f)

model_name = config.get("model_name")
if not model_name:
    raise ValueError(f"El archivo {MODEL_JSON_PATH} no contiene el nombre del modelo.")

# Construir la ruta completa del modelo ONNX
model_path = os.path.join(MODEL_DIR, model_name)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El modelo {model_name} no se encontró en {MODEL_DIR}.")

# Cargar el modelo ONNX
session = onnxruntime.InferenceSession(model_path)

# Definir esquema de entrada
class ModelInput(BaseModel):
    features: List[float]

# Iniciar la aplicación FastAPI
app = FastAPI()

@app.post("/predict")
def predict(input_data: ModelInput):
    input_array = [input_data.features]
    
    # Verificar dimensiones del input
    input_name = session.get_inputs()[0].name
    n_features_expected = session.get_inputs()[0].shape[1]
    if len(input_data.features) != n_features_expected:
        raise HTTPException(
            status_code=400,
            detail=f"El modelo espera {n_features_expected} características, pero se recibieron {len(input_data.features)}."
        )
    
    # Realizar predicción
    prediction = session.run(None, {input_name: input_array})[0][0]
    prediction_value = float(prediction) if isinstance(prediction, (int, float)) else prediction.tolist()
    return {"prediction CO in mg/m^3": prediction_value}

@app.get("/")
def read_root():
    return {"message": f"API para {model_name}"}
