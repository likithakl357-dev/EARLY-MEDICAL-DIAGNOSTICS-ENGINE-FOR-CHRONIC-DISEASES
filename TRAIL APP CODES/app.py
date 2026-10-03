import customtkinter as ctk
import pandas as pd
import joblib
import os


# ============================================================
# APPLICATION SETTINGS
# ============================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD TRAINED MACHINE LEARNING MODELS
# ============================================================

diabetes_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "medical_models",
        "diabetes_model.pkl"
    )
)

heart_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "medical_models",
        "heart_model.pkl"
    )
)

kidney_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "medical_models",
        "kidney_model.pkl"
    )
)


# ============================================================
# MAIN WINDOW
# ============================================================

app = ctk.CTk()

app.title(
    "Early Medical Diagnostics Engine"
)

app.geometry("1250x780")

app.minsize(1100, 700)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_risk():

    try:

        patient = {

            "age": float(age_entry.get()),

            "gender": gender_menu.get(),

            "bmi": float(bmi_entry.get()),

            "smoking_status":
                smoking_menu.get(),

            "alcohol_use":
                alcohol_menu.get(),

            "physical_activity":
                activity_menu.get(),

            "family_history_chronic_disease":
                family_menu.get(),

            "systolic_bp":
                float(systolic_entry.get()),

            "diastolic_bp":
                float(diastolic_entry.get()),

            "heart_rate":
                float(heart_rate_entry.get()),

            "glucose_mg_dl":
                float(glucose_entry.get()),

            "cholesterol_mg_dl":
                float(cholesterol_entry.get()),

            "hdl_mg_dl":
                float(hdl_entry.get()),

            "ldl_mg_dl":
                float(ldl_entry.get()),

            "triglycerides_mg_dl":
                float(triglycerides_entry.get()),

            "creatinine_mg_dl":
                float(creatinine_entry.get())
        }


        patient_df = pd.DataFrame([patient])


        # ====================================================
        # PREDICTIONS
        # ====================================================

        diabetes_prediction = diabetes_model.predict(
            patient_df
        )[0]

        heart_prediction = heart_model.predict(
            patient_df
        )[0]

        kidney_prediction = kidney_model.predict(
            patient_df
        )[0]


        # ====================================================
        # PREDICTED PROBABILITIES
        # ====================================================

        diabetes_probability = max(
            diabetes_model.predict_proba(
                patient_df
            )[0]
        ) * 100

        heart_probability = max(
            heart_model.predict_proba(
                patient_df
            )[0]
        ) * 100

        kidney_probability = max(
            kidney_model.predict_proba(
                patient_df
            )[0]
        ) * 100


        # ====================================================
        # OVERALL RISK
        # ====================================================

        risk_values = {
            "Low": 1,
            "Moderate": 2,
            "High": 3
        }

        highest_risk = max(
            risk_values[diabetes_prediction],
            risk_values[heart_prediction],
            risk_values[kidney_prediction]
        )


        if highest_risk == 3:

            overall = "HIGH RISK"

        elif highest_risk == 2:

            overall = "MODERATE RISK"

        else:

            overall = "LOW RISK"


        # ====================================================
        # UPDATE RESULTS
        # ====================================================

        diabetes_result.configure(
            text=(
                "🩸 DIABETES RISK\n\n"
                f"{diabetes_prediction.upper()}\n"
                f"Predicted Probability: "
                f"{diabetes_probability:.1f}%"
            )
        )


        heart_result.configure(
            text=(
                "❤️ HEART DISEASE RISK\n\n"
                f"{heart_prediction.upper()}\n"
                f"Predicted Probability: "
                f"{heart_probability:.1f}%"
            )
        )


        kidney_result.configure(
            text=(
                "🫘 KIDNEY DISEASE RISK\n\n"
                f"{kidney_prediction.upper()}\n"
                f"Predicted Probability: "
                f"{kidney_probability:.1f}%"
            )
        )


        overall_result.configure(
            text=(
                "OVERALL RISK\n\n"
                f"{overall}"
            )
        )


        status_label.configure(
            text="✓ Risk assessment completed successfully."
        )


    except ValueError:

        status_label.configure(
            text="⚠ Please enter valid numerical values.",
            text_color="red"
        )


    except Exception as e:

        status_label.configure(
            text=f"⚠ Error: {str(e)}",
            text_color="red"
        )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_fields():

    entries = [
        age_entry,
        bmi_entry,
        systolic_entry,
        diastolic_entry,
        heart_rate_entry,
        glucose_entry,
        cholesterol_entry,
        hdl_entry,
        ldl_entry,
        triglycerides_entry,
        creatinine_entry
    ]

    for entry in entries:

        entry.delete(
            0,
            "end"
        )


    gender_menu.set("Male")
    smoking_menu.set("Never")
    alcohol_menu.set("None")
    activity_menu.set("Moderate")
    family_menu.set("No")


    diabetes_result.configure(
        text="🩸 DIABETES RISK\n\nWaiting..."
    )

    heart_result.configure(
        text="❤️ HEART DISEASE RISK\n\nWaiting..."
    )

    kidney_result.configure(
        text="🫘 KIDNEY DISEASE RISK\n\nWaiting..."
    )

    overall_result.configure(
        text="OVERALL RISK\n\nWaiting..."
    )

    status_label.configure(
        text=""
    )


# ============================================================
# HEADER
# ============================================================

header = ctk.CTkFrame(
    app,
    corner_radius=0
)

header.pack(
    fill="x"
)


title_label = ctk.CTkLabel(
    header,
    text="🏥 EARLY MEDICAL DIAGNOSTICS ENGINE",
    font=ctk.CTkFont(
        size=28,
        weight="bold"
    )
)

title_label.pack(
    pady=(20, 5)
)


subtitle_label = ctk.CTkLabel(
    header,
    text="Chronic Disease Risk Assessment System",
    font=ctk.CTkFont(
        size=16
    )
)

subtitle_label.pack(
    pady=(0, 20)
)


# ============================================================
# MAIN FRAME
# ============================================================

main_frame = ctk.CTkFrame(
    app
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=20
)


# ============================================================
# LEFT COLUMN
# ============================================================

left_frame = ctk.CTkFrame(
    main_frame
)

left_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


ctk.CTkLabel(
    left_frame,
    text="👤 PATIENT INFORMATION",
    font=ctk.CTkFont(
        size=19,
        weight="bold"
    )
).pack(
    pady=15
)


age_entry = ctk.CTkEntry(
    left_frame,
    placeholder_text="Age"
)

age_entry.pack(
    padx=20,
    pady=7,
    fill="x"
)


gender_menu = ctk.CTkComboBox(
    left_frame,
    values=[
        "Male",
        "Female"
    ]
)

gender_menu.set("Male")

gender_menu.pack(
    padx=20,
    pady=7,
    fill="x"
)


bmi_entry = ctk.CTkEntry(
    left_frame,
    placeholder_text="BMI"
)

bmi_entry.pack(
    padx=20,
    pady=7,
    fill="x"
)


smoking_menu = ctk.CTkComboBox(
    left_frame,
    values=[
        "Never",
        "Former",
        "Current"
    ]
)

smoking_menu.set("Never")

smoking_menu.pack(
    padx=20,
    pady=7,
    fill="x"
)


alcohol_menu = ctk.CTkComboBox(
    left_frame,
    values=[
        "None",
        "Moderate",
        "Heavy"
    ]
)

alcohol_menu.set("None")

alcohol_menu.pack(
    padx=20,
    pady=7,
    fill="x"
)


activity_menu = ctk.CTkComboBox(
    left_frame,
    values=[
        "Low",
        "Moderate",
        "High"
    ]
)

activity_menu.set("Moderate")

activity_menu.pack(
    padx=20,
    pady=7,
    fill="x"
)


family_menu = ctk.CTkComboBox(
    left_frame,
    values=[
        "Yes",
        "No"
    ]
)

family_menu.set("No")

family_menu.pack(
    padx=20,
    pady=7,
    fill="x"
)


# ============================================================
# MIDDLE COLUMN
# ============================================================

middle_frame = ctk.CTkFrame(
    main_frame
)

middle_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


ctk.CTkLabel(
    middle_frame,
    text="🩺 MEDICAL PARAMETERS",
    font=ctk.CTkFont(
        size=19,
        weight="bold"
    )
).pack(
    pady=15
)


def create_entry(
    parent,
    placeholder
):

    entry = ctk.CTkEntry(
        parent,
        placeholder_text=placeholder
    )

    entry.pack(
        padx=20,
        pady=6,
        fill="x"
    )

    return entry


systolic_entry = create_entry(
    middle_frame,
    "Systolic BP"
)

diastolic_entry = create_entry(
    middle_frame,
    "Diastolic BP"
)

heart_rate_entry = create_entry(
    middle_frame,
    "Heart Rate"
)

glucose_entry = create_entry(
    middle_frame,
    "Glucose (mg/dL)"
)

cholesterol_entry = create_entry(
    middle_frame,
    "Cholesterol (mg/dL)"
)

hdl_entry = create_entry(
    middle_frame,
    "HDL (mg/dL)"
)

ldl_entry = create_entry(
    middle_frame,
    "LDL (mg/dL)"
)

triglycerides_entry = create_entry(
    middle_frame,
    "Triglycerides (mg/dL)"
)

creatinine_entry = create_entry(
    middle_frame,
    "Creatinine (mg/dL)"
)


# ============================================================
# RIGHT COLUMN
# ============================================================

right_frame = ctk.CTkFrame(
    main_frame
)

right_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


ctk.CTkLabel(
    right_frame,
    text="📊 RISK ASSESSMENT",
    font=ctk.CTkFont(
        size=19,
        weight="bold"
    )
).pack(
    pady=15
)


diabetes_result = ctk.CTkLabel(
    right_frame,
    text="🩸 DIABETES RISK\n\nWaiting...",
    font=ctk.CTkFont(
        size=16,
        weight="bold"
    ),
    height=100
)

diabetes_result.pack(
    padx=20,
    pady=8,
    fill="x"
)


heart_result = ctk.CTkLabel(
    right_frame,
    text="❤️ HEART DISEASE RISK\n\nWaiting...",
    font=ctk.CTkFont(
        size=16,
        weight="bold"
    ),
    height=100
)

heart_result.pack(
    padx=20,
    pady=8,
    fill="x"
)


kidney_result = ctk.CTkLabel(
    right_frame,
    text="🫘 KIDNEY DISEASE RISK\n\nWaiting...",
    font=ctk.CTkFont(
        size=16,
        weight="bold"
    ),
    height=100
)

kidney_result.pack(
    padx=20,
    pady=8,
    fill="x"
)


overall_result = ctk.CTkLabel(
    right_frame,
    text="OVERALL RISK\n\nWaiting...",
    font=ctk.CTkFont(
        size=21,
        weight="bold"
    ),
    height=120
)

overall_result.pack(
    padx=20,
    pady=12,
    fill="x"
)


# ============================================================
# BUTTONS
# ============================================================

button_frame = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

button_frame.pack(
    fill="x",
    padx=30,
    pady=5
)


predict_button = ctk.CTkButton(
    button_frame,
    text="🔍  ASSESS CHRONIC DISEASE RISK",
    command=predict_risk,
    height=50,
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    )
)

predict_button.pack(
    side="left",
    expand=True,
    fill="x",
    padx=5
)


clear_button = ctk.CTkButton(
    button_frame,
    text="↻  CLEAR",
    command=clear_fields,
    height=50,
    width=150
)

clear_button.pack(
    side="right",
    padx=5
)


# ============================================================
# STATUS
# ============================================================

status_label = ctk.CTkLabel(
    app,
    text=""
)

status_label.pack(
    pady=5
)


# ============================================================
# FOOTER
# ============================================================

footer = ctk.CTkLabel(
    app,
    text=(
        "For academic/research purposes only • "
        "Not a medical diagnosis"
    ),
    font=ctk.CTkFont(
        size=12
    )
)

footer.pack(
    pady=8
)


# ============================================================
# START APPLICATION
# ============================================================

app.mainloop()