import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "medical_models")

print("=" * 60)
print("KIDNEY MODEL DETAILED INSPECTION")
print("=" * 60)

# Load kidney model
kidney_path = os.path.join(MODEL_DIR, "kidney_model.pkl")
kidney_model = joblib.load(kidney_path)

print("\nMODEL TYPE:")
print(type(kidney_model))

print("\nMODEL CLASSES:")
print(kidney_model.classes_)

print("\nNUMBER OF CLASSES:")
print(len(kidney_model.classes_))

print("\nPIPELINE STEPS:")

if hasattr(kidney_model, "steps"):
    for name, step in kidney_model.steps:
        print(" -", name)
        print("   Type:", type(step))

print("\nMODEL DETAILS:")

# Check the final model inside the pipeline
if hasattr(kidney_model, "named_steps"):

    print("\nNamed steps:")
    for name in kidney_model.named_steps:
        print(" -", name)

    print("\nFinal model:")
    final_step = kidney_model.steps[-1][1]
    print(type(final_step))

    if hasattr(final_step, "n_features_in_"):
        print("\nFinal model expected features:")
        print(final_step.n_features_in_)

# Check feature names
if hasattr(kidney_model, "feature_names_in_"):
    print("\nFEATURE NAMES:")
    print(kidney_model.feature_names_in_)

print("\n" + "=" * 60)
print("KIDNEY MODEL INSPECTION COMPLETED")
print("=" * 60)

input("\nPress Enter to close...")