from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# cargar modelo
modelo = joblib.load("modelo_regresion.pkl")

# cargar dataset desde URL
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# INTERFAZ WEB
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard ML</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>

    <h1>Dashboard Viviendas 📊</h1>

    <label>Ingreso mínimo:</label>
    <input id="income" type="number" step="0.1">

    <button onclick="cargar()">Filtrar</button>

    <canvas id="grafico"></canvas>

    <h2>Predicción</h2>
    <input id="data" placeholder="Ej: 8.3,41,880,129,322,126,8.3,1">
    <button onclick="predecir()">Predecir</button>
    <p id="res"></p>

    <script>
    let chart;

    function cargar(){
        let income = document.getElementById("income").value;

        fetch("/data?income=" + income)
        .then(r=>r.json())
        .then(d=>{

            const puntos = d.x.map((x,i)=>({x:x, y:d.y[i]}));

            const ctx = document.getElementById('grafico');

            if(chart) chart.destroy();

            chart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Dataset real',
                        data: puntos
                    }]
                }
            });
        });
    }

    function predecir(){
        let valores = document.getElementById("data").value.split(",").map(Number);

        fetch("/predict", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body: JSON.stringify({input: valores})
        })
        .then(r=>r.json())
        .then(d=>{
            document.getElementById("res").innerText = "Predicción: " + d.prediccion;
        });
    }

    // carga inicial
    cargar();
    </script>

    </body>
    </html>
    """)

# DATOS PARA GRÁFICO
@app.route("/data")
def data():
    income = request.args.get("income")

    data = df.copy()

    if income:
        data = data[data["median_income"] > float(income)]

    return jsonify({
        "x": data["median_income"].tolist(),
        "y": data["median_house_value"].tolist()
    })

# PREDICCIÓN
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    valores = np.array(data["input"]).reshape(1, -1)
    pred = modelo.predict(valores)

    return jsonify({"prediccion": float(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
