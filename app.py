"""
app.py
------
Flask web server that serves the ML-powered insulin calculator.
Loads trained models and exposes a /predict API endpoint.

Run:
    python app.py
Then open:
    http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load trained models (run train_models.py first!)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

try:
    dose_model  = joblib.load(os.path.join(MODEL_DIR, 'insulin_dose_model.pkl'))
    bg_model    = joblib.load(os.path.join(MODEL_DIR, 'bg_prediction_model.pkl'))
    hypo_model  = joblib.load(os.path.join(MODEL_DIR, 'hypo_risk_model.pkl'))
    FEATURES    = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✅ All models loaded successfully.")
except FileNotFoundError:
    print("❌ Models not found! Please run train_models.py first.")
    dose_model = bg_model = hypo_model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body:
    {
        "carbs_g": 60,
        "current_bg": 180,
        "time_of_day": 12,
        "activity_level": 1,
        "weight_kg": 70,
        "icr": 10,
        "cf": 50,
        "target_bg": 100
    }
    Returns predicted insulin dose, post-meal BG, and hypo risk.
    """
    if dose_model is None:
        return jsonify({"error": "Models not loaded. Run train_models.py first."}), 500

    try:
        data = request.get_json()

        # Build feature array in correct order
        features = np.array([[
            float(data['carbs_g']),
            float(data['current_bg']),
            float(data['time_of_day']),
            float(data['activity_level']),
            float(data['weight_kg']),
            float(data['icr']),
            float(data['cf']),
            float(data['target_bg']),
        ]])

        # ── Run predictions
        insulin_dose = float(dose_model.predict(features)[0])
        bg_after_2h  = float(bg_model.predict(features)[0])
        hypo_risk_id = int(hypo_model.predict(features)[0])
        hypo_proba   = hypo_model.predict_proba(features)[0].tolist()

        risk_labels = ['Low', 'Medium', 'High']
        risk_colors = ['#58d5a0', '#f7c948', '#f87171']

        return jsonify({
            "insulin_dose":   round(insulin_dose, 2),
            "bg_after_2h":    round(bg_after_2h, 1),
            "hypo_risk":      risk_labels[hypo_risk_id],
            "hypo_risk_color": risk_colors[hypo_risk_id],
            "hypo_proba": {
                "low":    round(hypo_proba[0] * 100, 1),
                "medium": round(hypo_proba[1] * 100, 1),
                "high":   round(hypo_proba[2] * 100, 1),
            }
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
