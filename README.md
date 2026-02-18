# 🩺 Diabetes Insulin Dose Predictor — Machine Learning Project

A machine learning web application that predicts insulin doses for diabetic patients based on carbohydrate intake, blood glucose, activity level, and personal therapy settings.

> ⚠️ **Educational purposes only.** Not for real medical use.

---

## 🤖 ML Models Used

| Model | Algorithm | Task |
|-------|-----------|------|
| Insulin Dose Predictor | Random Forest Regressor | How many units of insulin to inject |
| Post-Meal BG Predictor | Gradient Boosting Regressor | Predicted blood glucose 2h after meal |
| Hypo Risk Classifier | Random Forest Classifier | Low / Medium / High hypoglycemia risk |

---

## 📁 Project Structure

```
diabetes_ml/
├── data/
│   ├── generate_data.py       ← Generates synthetic training data
│   └── diabetes_data.csv      ← Generated dataset (auto-created)
├── model/
│   ├── insulin_dose_model.pkl ← Trained RF Regressor
│   ├── bg_prediction_model.pkl← Trained GB Regressor
│   └── hypo_risk_model.pkl    ← Trained RF Classifier
├── templates/
│   └── index.html             ← Web UI (served by Flask)
├── train_models.py            ← Train & evaluate all 3 models
├── app.py                     ← Flask API + web server
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_models.py
```
This will:
- Generate synthetic patient data (2000 samples)
- Train 3 ML models
- Print accuracy metrics
- Save `.pkl` model files

### 3. Start the web app
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 📊 Features Used by Models

| Feature | Description |
|---------|-------------|
| `carbs_g` | Grams of carbohydrates in meal |
| `current_bg` | Blood glucose before meal (mg/dL) |
| `time_of_day` | Hour of day (0–23) |
| `activity_level` | 0=Sedentary, 1=Light, 2=Moderate, 3=Intense |
| `weight_kg` | Patient weight |
| `icr` | Insulin-to-carb ratio (g per unit) |
| `cf` | Correction factor (mg/dL drop per unit) |
| `target_bg` | Patient's target blood glucose |

---

## 📈 Model Performance (on test set)

After training you'll see metrics like:
- **Dose Model MAE**: ~0.3 units
- **BG Model MAE**: ~15 mg/dL  
- **Hypo Classifier Accuracy**: ~92%

---

## 🔬 Real Datasets to Try

Replace the synthetic data with real diabetes datasets:
- [OhioT1DM Dataset](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) — CGM + insulin pump data
- [Pima Indians Diabetes (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [UCI Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes)

---

## 🛠 Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML models
- **pandas / numpy** — Data processing
- **Flask** — Web API
- **Joblib** — Model serialization

---

## 👨‍💻 Author

Built as a machine learning portfolio project for diabetes management research.
