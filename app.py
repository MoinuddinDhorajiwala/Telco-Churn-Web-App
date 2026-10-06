from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import os

from preprocess import preprocess_input

app = Flask(__name__)

# Allow requests from the Netlify frontend
CORS(
    app,
    resources={
        r"/predict": {
            "origins": "https://telecom-churn-bda.netlify.app"
        }
    }
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "Churn-Classification.joblib"
)

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Churn Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received."
            }), 400

        processed_data = preprocess_input(data)

        prediction = model.predict(processed_data)[0]

        probability = model.predict_proba(
            processed_data
        )[0][1]

        if prediction == 1:
            result = "Churn"
        else:
            result = "No Churn"

        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return jsonify({
            "prediction": result,
            "churn_probability": round(
                float(probability), 4
            ),
            "risk": risk
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )