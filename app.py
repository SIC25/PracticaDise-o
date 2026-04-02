from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load("modelo_regresion.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        valores = np.array(data["input"]).reshape(1, -1)
        pred = modelo.predict(valores)
        return jsonify({"prediccion": float(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)