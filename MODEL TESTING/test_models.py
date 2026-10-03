import pandas as pd
import joblib


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_FILE = "chronic_disease_risk_dataset (1) (1).csv"

df = pd.read_csv(DATASET_FILE)


# ============================================================
# LOAD NEW MODELS
# ============================================================

diabetes_model = joblib.load(
    "medical_models/diabetes_model.pkl"
)

heart_model = joblib.load(
    "medical_models/heart_model.pkl"
)

kidney_model = joblib.load(
    "medical_models/kidney_model.pkl"
)


# ============================================================
# REMOVE TARGET COLUMNS
# ============================================================

DROP_COLUMNS = [
    "patient_id",
    "diabetes_risk",
    "heart_disease_risk",
    "kidney_disease_risk"
]

X = df.drop(columns=DROP_COLUMNS)


# ============================================================
# TEST FIRST 10 PATIENTS
# ============================================================

test_data = X.iloc[:10]


print("=" * 70)
print("TESTING NEW INDEPENDENT MODELS")
print("=" * 70)


# ============================================================
# DIABETES
# ============================================================

diabetes_predictions = diabetes_model.predict(test_data)

print("\nDIABETES PREDICTIONS")

for i, prediction in enumerate(diabetes_predictions, start=1):
    print(f"Patient {i}: {prediction}")


# ============================================================
# HEART DISEASE
# ============================================================

heart_predictions = heart_model.predict(test_data)

print("\nHEART DISEASE PREDICTIONS")

for i, prediction in enumerate(heart_predictions, start=1):
    print(f"Patient {i}: {prediction}")


# ============================================================
# KIDNEY DISEASE
# ============================================================

kidney_predictions = kidney_model.predict(test_data)

print("\nKIDNEY DISEASE PREDICTIONS")

for i, prediction in enumerate(kidney_predictions, start=1):
    print(f"Patient {i}: {prediction}")


# ============================================================
# KIDNEY PROBABILITIES
# ============================================================

kidney_probabilities = kidney_model.predict_proba(test_data)

print("\nKIDNEY PREDICTION PROBABILITIES")

print("Class order:")
print(kidney_model.classes_)

for i, probabilities in enumerate(
    kidney_probabilities,
    start=1
):
    print(
        f"Patient {i}: {probabilities}"
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("NEW MODEL TEST COMPLETED")
print("=" * 70)