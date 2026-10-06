from flask import Flask, render_template, request, jsonify
import joblib
import os

from preprocess import preprocess_input


app = Flask(__name__)


# ============================================================
# Load trained model
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "Churn-Classification.joblib"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "Churn Prediction API is running"
    })


# ============================================================
# Prediction API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON data from frontend
        # ----------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received."
            }), 400


        # ----------------------------------------------------
        # Preprocess input
        # ----------------------------------------------------

        processed_data = preprocess_input(data)


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(processed_data)[0]


        # ----------------------------------------------------
        # Get churn probability
        # ----------------------------------------------------

        probability = model.predict_proba(
            processed_data
        )[0][1]


        # ----------------------------------------------------
        # Convert prediction to readable result
        # ----------------------------------------------------

        if prediction == 1:

            result = "Churn"

        else:

            result = "No Churn"


        # ----------------------------------------------------
        # Determine risk level
        # ----------------------------------------------------

        if probability >= 0.70:

            risk = "High"

        elif probability >= 0.40:

            risk = "Medium"

        else:

            risk = "Low"


        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        return jsonify({

            "prediction": result,

            "churn_probability":
                round(float(probability), 4),

            "risk": risk

        })


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )