"""
train_model.py
--------------
Machine Learning pipeline for Airline Price Prediction.
Trains a Multiple Linear Regression model on flights.csv
and serialises it to model.pkl.

Run this ONCE before starting the Flask app:
    python train_model.py
"""

import re
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows from '{filepath}'.")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING HELPERS
# ─────────────────────────────────────────────

def parse_duration(duration_str: str) -> int:
    """
    Convert a duration string like '2h 50m', '3h', or '45m'
    into total minutes (integer).
    """
    if pd.isna(duration_str):
        return 0
    duration_str = str(duration_str).strip()
    hours   = re.search(r'(\d+)\s*h', duration_str)
    minutes = re.search(r'(\d+)\s*m', duration_str)
    h = int(hours.group(1))   if hours   else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m


def extract_time_parts(time_str: str):
    """
    Extract hour and minute from a time string like '22:20' or '01:10 22 Mar'.
    Returns (hour: int, minute: int).
    """
    if pd.isna(time_str):
        return 0, 0
    match = re.search(r'(\d{1,2}):(\d{2})', str(time_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def encode_stops(stops_str: str) -> int:
    """
    Map 'non-stop' → 0, '1 stop' → 1, '2 stops' → 2, etc.
    Defaults to 0 for unrecognised values.
    """
    if pd.isna(stops_str):
        return 0
    s = str(stops_str).lower().strip()
    if 'non' in s:
        return 0
    match = re.search(r'(\d+)', s)
    return int(match.group(1)) if match else 0


# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Clean and engineer features from the raw DataFrame.
    Returns X (feature matrix) and y (target vector).
    """
    df = df.copy()

    # Drop rows with missing target
    df.dropna(subset=['price'], inplace=True)

    # Fill remaining NaNs in string columns
    for col in ['additional_info', 'route']:
        if col in df.columns:
            df[col] = df[col].fillna('No info')

    # ── Duration → minutes ──────────────────
    df['duration_min'] = df['duration'].apply(parse_duration)

    # ── Departure time features ─────────────
    df['dep_hour'], df['dep_min'] = zip(*df['dep_time'].apply(extract_time_parts))

    # ── Arrival time features ────────────────
    df['arr_hour'], df['arr_min'] = zip(*df['arrival_time'].apply(extract_time_parts))

    # ── Journey date features ────────────────
    try:
        df['journey_date'] = pd.to_datetime(df['date_of_journey'], dayfirst=True, errors='coerce')
        df['journey_day']   = df['journey_date'].dt.day
        df['journey_month'] = df['journey_date'].dt.month
    except Exception:
        df['journey_day']   = 1
        df['journey_month'] = 1

    # ── Stops ────────────────────────────────
    df['num_stops'] = df['total_stops'].apply(encode_stops)

    # ── Select final features ────────────────
    cat_features  = ['airline', 'source', 'destination']
    num_features  = [
        'duration_min', 'dep_hour', 'dep_min',
        'arr_hour', 'arr_min',
        'journey_day', 'journey_month',
        'num_stops'
    ]

    X = df[cat_features + num_features]
    y = df['price'].astype(float)

    return X, y, cat_features, num_features


# ─────────────────────────────────────────────
# 4. BUILD & TRAIN MODEL
# ─────────────────────────────────────────────

def build_pipeline(cat_features: list, num_features: list) -> Pipeline:
    """
    Construct a scikit-learn Pipeline:
      - OneHotEncoder for categorical columns
      - LinearRegression estimator
    """
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', 'passthrough', num_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor',    LinearRegression())
    ])
    return pipeline


def train_and_evaluate(pipeline: Pipeline, X, y):
    """Split data, fit the pipeline, print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print(f"[METRICS] MAE  : ₹{mae:,.2f}")
    print(f"[METRICS] R²   : {r2:.4f}")
    return pipeline


# ─────────────────────────────────────────────
# 5. SAVE MODEL ARTEFACT
# ─────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str = 'model.pkl'):
    """Serialise the trained pipeline to disk with pickle."""
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[INFO] Model saved → '{path}'")


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    df                             = load_data('flights.csv')
    X, y, cat_features, num_feats  = preprocess(df)
    pipeline                       = build_pipeline(cat_features, num_feats)
    pipeline                       = train_and_evaluate(pipeline, X, y)
    save_model(pipeline, 'model.pkl')
    print("[DONE] Training complete.")
