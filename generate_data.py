"""
generate_data.py
----------------
Generates synthetic but realistic diabetes patient data for training our ML models.
In a real project, you would replace this with actual patient data (e.g. OhioT1DM dataset).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_dataset(n_samples=2000):
    """
    Simulate patient meal + glucose records.
    Features:
        - carbs_g         : grams of carbohydrates eaten
        - current_bg      : blood glucose before meal (mg/dL)
        - time_of_day     : hour of day (0-23)
        - activity_level  : 0=sedentary, 1=light, 2=moderate, 3=intense
        - weight_kg       : patient weight
        - icr             : insulin-to-carb ratio (patient-specific)
        - cf              : correction factor (patient-specific)
        - target_bg       : patient's target blood glucose

    Targets:
        - insulin_dose    : correct insulin units to give (regression)
        - bg_after_2h     : blood glucose 2 hours after meal (regression)
        - hypo_risk       : 0=low risk, 1=medium risk, 2=high risk (classification)
    """

    # ── Patient-specific constants (vary per record to simulate multiple patients)
    weight_kg = np.random.normal(75, 15, n_samples).clip(45, 130)
    icr = np.random.normal(10, 3, n_samples).clip(5, 20)       # grams per unit
    cf  = np.random.normal(50, 12, n_samples).clip(20, 100)    # mg/dL per unit
    target_bg = np.random.normal(100, 10, n_samples).clip(80, 130)

    # ── Input features
    carbs_g        = np.random.uniform(10, 130, n_samples)
    current_bg     = np.random.normal(140, 50, n_samples).clip(50, 400)
    time_of_day    = np.random.randint(0, 24, n_samples)
    activity_level = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])

    # ── Activity multiplier: exercise increases insulin sensitivity → less insulin needed
    activity_multiplier = 1 - (activity_level * 0.08)

    # ── Meal dose (carbs / ICR) with activity adjustment
    meal_dose = (carbs_g / icr) * activity_multiplier

    # ── Correction dose (only if BG is above target)
    correction_dose = np.maximum((current_bg - target_bg) / cf, 0)

    # ── Total insulin dose + realistic noise
    insulin_dose = (meal_dose + correction_dose) + np.random.normal(0, 0.3, n_samples)
    insulin_dose = insulin_dose.clip(0, 30)

    # ── Post-meal BG prediction (simplified physiology model)
    # BG rises from carbs, drops from insulin, affected by activity
    bg_rise       = carbs_g * 4                                       # ~4 mg/dL per gram carb
    bg_drop       = insulin_dose * cf                                  # insulin brings BG down
    activity_drop = activity_level * 15                                # exercise drops BG
    bg_after_2h   = current_bg + bg_rise - bg_drop - activity_drop
    bg_after_2h  += np.random.normal(0, 15, n_samples)                # biological noise
    bg_after_2h   = bg_after_2h.clip(40, 500)

    # ── Hypoglycemia risk label  (<70 = high risk, 70-100 = medium, >100 = low)
    hypo_risk = np.where(bg_after_2h < 70, 2,
                np.where(bg_after_2h < 100, 1, 0))

    df = pd.DataFrame({
        'carbs_g':        carbs_g.round(1),
        'current_bg':     current_bg.round(1),
        'time_of_day':    time_of_day,
        'activity_level': activity_level,
        'weight_kg':      weight_kg.round(1),
        'icr':            icr.round(1),
        'cf':             cf.round(1),
        'target_bg':      target_bg.round(1),
        # ── Targets
        'insulin_dose':   insulin_dose.round(2),
        'bg_after_2h':    bg_after_2h.round(1),
        'hypo_risk':      hypo_risk,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset(2000)
    df.to_csv("diabetes_data.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} rows")
    print(df.head())
    print("\nTarget distribution - Hypo Risk:")
    print(df['hypo_risk'].value_counts().rename({0:'Low', 1:'Medium', 2:'High'}))
