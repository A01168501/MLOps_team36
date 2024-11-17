import sys
import os
import json
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    # Recibir parámetros desde la línea de comandos
    if len(sys.argv) != 4:
        print("Uso: python convert_to_onnx.py <modelo.pkl> <directorio_onnx> <model.json>")
        sys.exit(1)

    model_pkl_path = sys.argv[1]
    onnx_dir = sys.argv[2]
    json_path = sys.argv[3]

    # Cargar el modelo
    if not os.path.exists(model_pkl_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_pkl_path}.")
    model = joblib.load(model_pkl_path)

    # Determinar el número de características automáticamente
    if hasattr(model, "n_features_in_"):
        n_features = model.n_features_in_
    elif hasattr(model, "coef_") and model.coef_ is not None:
        n_features = model.coef_.shape[1]
    else:
        raise ValueError("No se pudo determinar el número de características del modelo.")

    print(f"El modelo espera {n_features} características.")

    # Convertir el modelo a ONNX
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Crear directorio si no existe
    os.makedirs(onnx_dir, exist_ok=True)

    # Guardar el modelo ONNX
    onnx_file_name = os.path.basename(model_pkl_path).replace(".pkl", ".onnx")
    onnx_file_path = os.path.join(onnx_dir, onnx_file_name)
    with open(onnx_file_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Modelo ONNX guardado en {onnx_file_path}")

    # Actualizar el archivo JSON
    model_metadata = {
        "model_name": onnx_file_name
    }
    with open(json_path, "w") as f:
        json.dump(model_metadata, f, indent=4)
    print(f"Archivo JSON actualizado en {json_path}")

if __name__ == "__main__":
    main()
