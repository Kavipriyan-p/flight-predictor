"""
app.py
------
Flask web application for Airline Ticket Price Prediction.

Routes:
    GET  /          → Landing / prediction form
    POST /predict   → Run ML model and return predicted price

Usage:
    python app.py
"""

import os
import re
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────
# APP INITIALISATION
# ─────────────────────────────────────────────

app = Flask(__name__)

# ── Load model once at startup ───────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def load_model():
    """Load the pickled pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run 'python train_model.py' first."
        )
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    MODEL = load_model()
    print("[INFO] Model loaded successfully.")
except FileNotFoundError as e:
    MODEL = None
    print(f"[WARN] {e}")


# ─────────────────────────────────────────────
# HELPER: MIRROR OF train_model.py TRANSFORMERS
# (kept here so we don't import train_model.py)
# ─────────────────────────────────────────────

def parse_duration(duration_str: str) -> int:
    """Convert '2h 30m' → 150 minutes."""
    if not duration_str:
        return 0
    hours   = re.search(r'(\d+)\s*h', str(duration_str))
    minutes = re.search(r'(\d+)\s*m', str(duration_str))
    return (int(hours.group(1)) if hours else 0) * 60 + \
           (int(minutes.group(1)) if minutes else 0)


def encode_stops(stops_str: str) -> int:
    """Convert stop description to integer count."""
    if not stops_str:
        return 0
    s = str(stops_str).lower()
    if 'non' in s:
        return 0
    match = re.search(r'(\d+)', s)
    return int(match.group(1)) if match else 0


def build_input_df(form) -> pd.DataFrame:
    """
    Parse form data and construct a single-row DataFrame
    matching the training feature schema.
    """
    airline     = form.get('airline',     'IndiGo')
    source      = form.get('source',      'Delhi')
    destination = form.get('destination', 'Mumbai')
    dep_time    = form.get('dep_time',    '08:00')
    arr_time    = form.get('arr_time',    '10:00')
    duration    = form.get('duration',    '2h 0m')
    stops       = form.get('stops',       'non-stop')
    journey_date= form.get('journey_date','2019-06-15')

    # Parse departure time
    dep_h, dep_m = 8, 0
    dm = re.match(r'(\d{1,2}):(\d{2})', dep_time)
    if dm:
        dep_h, dep_m = int(dm.group(1)), int(dm.group(2))

    # Parse arrival time
    arr_h, arr_m = 10, 0
    am = re.match(r'(\d{1,2}):(\d{2})', arr_time)
    if am:
        arr_h, arr_m = int(am.group(1)), int(am.group(2))

    # Parse journey date
    day, month = 15, 6
    try:
        jd = pd.to_datetime(journey_date, errors='coerce')
        if jd is not pd.NaT:
            day, month = jd.day, jd.month
    except Exception:
        pass

    duration_min = parse_duration(duration)
    num_stops    = encode_stops(stops)

    row = {
        'airline':      airline,
        'source':       source,
        'destination':  destination,
        'duration_min': duration_min,
        'dep_hour':     dep_h,
        'dep_min':      dep_m,
        'arr_hour':     arr_h,
        'arr_min':      arr_m,
        'journey_day':  day,
        'journey_month':month,
        'num_stops':    num_stops,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Render the landing / prediction form page."""
    return render_template('index.html', prediction=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept form submission, run the ML model,
    and return the predicted price back to the template.
    """
    prediction = None
    error      = None

    if MODEL is None:
        error = "Model is not loaded. Please run 'python train_model.py' first."
        return render_template('index.html', prediction=prediction, error=error)

    try:
        input_df   = build_input_df(request.form)
        raw_price  = MODEL.predict(input_df)[0]

        # Clip to a realistic minimum (cost floor) and round
        predicted  = max(float(raw_price), 999.0)
        prediction = f"₹{predicted:,.0f}"

    except ValueError as ve:
        error = f"Invalid input: {ve}"
    except Exception:
        error = "An unexpected error occurred. Please check your inputs."
        app.logger.error(traceback.format_exc())

    return render_template('index.html', prediction=prediction, error=error)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, port=5000)
