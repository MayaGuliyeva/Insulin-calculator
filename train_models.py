"""
train_models.py
---------------
Trains three machine learning models for the diabetes insulin calculator:

  1. insulin_dose_model   → Random Forest Regressor  (predicts units of insulin)
  2. bg_prediction_model  → Gradient Boosting Regressor (predicts BG after 2h)
  3. hypo_risk_model      → Random Forest Classifier (Low / Medium / High risk)

Run this script once to generate the saved model files (.pkl).
Then app.py will load them to serve predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, accuracy_score)
from sklearn.pipeline import Pipeline

# ── Add parent path so we can import generate_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from generate_data import generate_dataset


# ════════════════════════════════════════════════
#  0. Load or Generate Data
# ════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'diabetes_data.csv')

if os.path.exists(DATA_PATH):
    print("📂 Loading existing dataset...")
    df = pd.read_csv(DATA_PATH)
else:
    print("🔄 Generating synthetic dataset...")
    df = generate_dataset(2000)
    df.to_csv(DATA_PATH, index=False)

print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {list(df.columns)}\n")


# ════════════════════════════════════════════════
#  1. Feature / Target Split
# ════════════════════════════════════════════════
FEATURES = ['carbs_g', 'current_bg', 'time_of_day', 'activity_level',
            'weight_kg', 'icr', 'cf', 'target_bg']

X = df[FEATURES]
y_dose  = df['insulin_dose']    # regression target 1
y_bg    = df['bg_after_2h']     # regression target 2
y_hypo  = df['hypo_risk']       # classification target (0/1/2)

X_train, X_test, yd_train, yd_test, yb_train, yb_test, yh_train, yh_test = \
    train_test_split(X, y_dose, y_bg, y_hypo, test_size=0.2, random_state=42)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}\n")


# ════════════════════════════════════════════════
#  2. Model 1 — Insulin Dose Prediction
#     Random Forest Regressor
# ════════════════════════════════════════════════
print("=" * 50)
print("MODEL 1: Insulin Dose Prediction")
print("=" * 50)

dose_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

dose_pipeline.fit(X_train, yd_train)
yd_pred = dose_pipeline.predict(X_test)

mae_dose = mean_absolute_error(yd_test, yd_pred)
r2_dose  = r2_score(yd_test, yd_pred)
print(f"  Mean Absolute Error : {mae_dose:.3f} units")
print(f"  R² Score            : {r2_dose:.4f}")

# Feature importance
importances = dose_pipeline.named_steps['model'].feature_importances_
feat_imp = sorted(zip(FEATURES, importances), key=lambda x: -x[1])
print("\n  Feature Importances:")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 40)
    print(f"    {feat:<18} {bar} {imp:.3f}")


# ════════════════════════════════════════════════
#  3. Model 2 — Post-Meal Blood Glucose Prediction
#     Gradient Boosting Regressor
# ════════════════════════════════════════════════
print("\n" + "=" * 50)
print("MODEL 2: Post-Meal BG Prediction (2h)")
print("=" * 50)

bg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])

bg_pipeline.fit(X_train, yb_train)
yb_pred = bg_pipeline.predict(X_test)

mae_bg = mean_absolute_error(yb_test, yb_pred)
r2_bg  = r2_score(yb_test, yb_pred)
print(f"  Mean Absolute Error : {mae_bg:.1f} mg/dL")
print(f"  R² Score            : {r2_bg:.4f}")


# ════════════════════════════════════════════════
#  4. Model 3 — Hypoglycemia Risk Classification
#     Random Forest Classifier
# ════════════════════════════════════════════════
print("\n" + "=" * 50)
print("MODEL 3: Hypoglycemia Risk Classification")
print("=" * 50)

hypo_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

hypo_pipeline.fit(X_train, yh_train)
yh_pred = hypo_pipeline.predict(X_test)

acc = accuracy_score(yh_test, yh_pred)
print(f"  Accuracy: {acc:.4f}")
print("\n  Classification Report:")
print(classification_report(yh_test, yh_pred,
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))


# ════════════════════════════════════════════════
#  5. Save Models
# ════════════════════════════════════════════════
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(dose_pipeline, os.path.join(MODEL_DIR, 'insulin_dose_model.pkl'))
joblib.dump(bg_pipeline,   os.path.join(MODEL_DIR, 'bg_prediction_model.pkl'))
joblib.dump(hypo_pipeline, os.path.join(MODEL_DIR, 'hypo_risk_model.pkl'))
joblib.dump(FEATURES,      os.path.join(MODEL_DIR, 'feature_names.pkl'))

print("\n✅ All models saved to /model/")
print("   → insulin_dose_model.pkl")
print("   → bg_prediction_model.pkl")
print("   → hypo_risk_model.pkl")
print("\n🚀 Now run:  python app.py")
