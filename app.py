from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load("modelo_regresion.pkl")

# Página principal (interfaz bonita)
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
        <h1>Predicción de Viviendas 🏠</h1>

        <input id="data" placeholder="Ej: 8.3,41,880,129,322,126,8.3,1">
        <button onclick="predecir()">Predecir</button>

        <h2 id="resultado"></h2>

        <canvas id="grafico" width="400" height="200"></canvas>

        <script>
        let chart;

        function predecir(){
            let valores = document.getElementById("data").value.split(",").map(Number);

            fetch("/predict", {
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body: JSON.stringify({input: valores})
            })
            .then(r=>r.json())
            .then(d=>{
                document.getElementById("resultado").innerText = "Predicción: " + d.prediccion;

                // gráfico
                const ctx = document.getElementById('grafico').getContext('2d');

                if(chart) chart.destroy();

                chart = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Punto ingresado',
                            data: [{x: valores[0], y: d.prediccion}]
                        }]
                    }
                });
            });
        }
        </script>
    </body>
    </html>
    """)

# Predicción
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    valores = np.array(data["input"]).reshape(1, -1)
    pred = modelo.predict(valores)

    return jsonify({
        "prediccion": float(pred[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
