
from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# comments: Robust model path for Render/Codespaces
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "wine_quality_model.joblib")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # comments: Read input values in the same order expected by the model
        values = [float(request.form[f]) for f in FEATURES]
        X = np.array(values).reshape(1, -1)
        pred = float(model.predict(X)[0])
        prediction = round(pred, 2)

    return render_template("index.html", prediction=prediction, features=FEATURES)

# comments: Render uses gunicorn, not app.run()
