import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "chronic_disease_risk_dataset (1) (1).csv"

MODEL_FOLDER = "medical_models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING MEDICAL DATASET")
print("=" * 70)

df = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully.")
print("Number of patients:", len(df))
print("Number of columns:", len(df.columns))

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# TARGET COLUMNS
# ============================================================

TARGET_COLUMNS = [
    "diabetes_risk",
    "heart_disease_risk",
    "kidney_disease_risk"
]


# ============================================================
# REMOVE TARGET COLUMNS AND PATIENT ID FROM INPUT FEATURES
# ============================================================

# These columns are NOT allowed to be input features.
# They are the answers that the models are supposed to predict.

DROP_COLUMNS = [
    "patient_id",
    "diabetes_risk",
    "heart_disease_risk",
    "kidney_disease_risk"
]


# Check that all required columns exist
for column in TARGET_COLUMNS:
    if column not in df.columns:
        raise ValueError(
            f"Target column '{column}' was not found in the dataset."
        )


# Input features
X = df.drop(columns=DROP_COLUMNS)


print("\n" + "=" * 70)
print("INPUT FEATURES")
print("=" * 70)

print("\nFeatures used by ALL three models:")

for feature in X.columns:
    print("-", feature)


print("\nTotal input columns:", len(X.columns))


# ============================================================
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# FUNCTION TO TRAIN ONE MODEL
# ============================================================

def train_model(target_column, model_filename):

    print("\n")
    print("=" * 70)
    print("TRAINING MODEL:", target_column)
    print("=" * 70)

    # Target / answer
    y = df[target_column]

    print("\nTarget distribution:")
    print(y.value_counts())

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Complete pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                random_forest
            )
        ]
    )

    # Train
    print("\nTraining...")
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:")
    print(f"{accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Save model
    model_path = os.path.join(
        MODEL_FOLDER,
        model_filename
    )

    joblib.dump(model, model_path)

    print("Model saved:")
    print(model_path)

    return model


# ============================================================
# TRAIN THREE INDEPENDENT MODELS
# ============================================================

diabetes_model = train_model(
    "diabetes_risk",
    "diabetes_model.pkl"
)

heart_model = train_model(
    "heart_disease_risk",
    "heart_model.pkl"
)

kidney_model = train_model(
    "kidney_disease_risk",
    "kidney_model.pkl"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nThree independent models have been created:")

print("1. diabetes_model.pkl")
print("2. heart_model.pkl")
print("3. kidney_model.pkl")

print("\nSaved inside:")
print(os.path.abspath(MODEL_FOLDER))

print("\nIMPORTANT:")
print("The three disease-risk columns were excluded from ALL input features.")

print("\nNo target leakage is being used.")

print("\n" + "=" * 70)