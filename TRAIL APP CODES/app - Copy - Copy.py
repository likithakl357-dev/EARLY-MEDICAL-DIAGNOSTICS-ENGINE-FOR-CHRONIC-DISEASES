import customtkinter as ctk
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from tkinter import filedialog, messagebox


# ============================================================
# CONFIGURATION
# ============================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD MODELS
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
# MAIN APPLICATION
# ============================================================

app = ctk.CTk()

app.title(
    "Early Medical Diagnostics Engine"
)

app.geometry(
    "1400x850"
)

app.minsize(
    1200,
    750
)


# ============================================================
# COLORS
# ============================================================

SIDEBAR_COLOR = "#172033"
CARD_COLOR = "#FFFFFF"
BACKGROUND_COLOR = "#F4F7FB"
TEXT_COLOR = "#172033"
SECONDARY_TEXT = "#64748B"
BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
RED = "#DC2626"


# ============================================================
# FUNCTIONS
# ============================================================

def show_dashboard():

    clear_content()

    title = ctk.CTkLabel(
        content,
        text="Dashboard",
        font=ctk.CTkFont(
            size=30,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    )

    title.pack(
        anchor="w",
        padx=30,
        pady=(25, 5)
    )


    subtitle = ctk.CTkLabel(
        content,
        text="Early detection and chronic disease risk assessment",
        font=ctk.CTkFont(size=15),
        text_color=SECONDARY_TEXT
    )

    subtitle.pack(
        anchor="w",
        padx=30
    )


    # --------------------------------------------------------
    # STAT CARDS
    # --------------------------------------------------------

    cards = ctk.CTkFrame(
        content,
        fg_color="transparent"
    )

    cards.pack(
        fill="x",
        padx=25,
        pady=30
    )


    create_dashboard_card(
        cards,
        "🩸",
        "Diabetes",
        "ML Risk Assessment",
        BLUE,
        0
    )

    create_dashboard_card(
        cards,
        "❤️",
        "Heart Disease",
        "ML Risk Assessment",
        RED,
        1
    )

    create_dashboard_card(
        cards,
        "🫘",
        "Kidney Disease",
        "ML Risk Assessment",
        GREEN,
        2
    )


    # --------------------------------------------------------
    # WELCOME CARD
    # --------------------------------------------------------

    welcome = ctk.CTkFrame(
        content,
        fg_color=CARD_COLOR,
        corner_radius=18
    )

    welcome.pack(
        fill="x",
        padx=30,
        pady=10
    )


    ctk.CTkLabel(
        welcome,
        text="Welcome to Early Medical Diagnostics",
        font=ctk.CTkFont(
            size=24,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        anchor="w",
        padx=30,
        pady=(25, 8)
    )


    ctk.CTkLabel(
        welcome,
        text=(
            "This system uses trained machine-learning models "
            "to assess chronic disease risk based on patient "
            "health parameters."
        ),
        font=ctk.CTkFont(size=15),
        text_color=SECONDARY_TEXT,
        justify="left"
    ).pack(
        anchor="w",
        padx=30,
        pady=(0, 25)
    )


    # --------------------------------------------------------
    # QUICK ACTION
    # --------------------------------------------------------

    quick = ctk.CTkButton(
        welcome,
        text="🔬  Start New Risk Assessment",
        height=48,
        width=260,
        font=ctk.CTkFont(
            size=15,
            weight="bold"
        ),
        command=show_assessment
    )

    quick.pack(
        anchor="w",
        padx=30,
        pady=(0, 30)
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    ctk.CTkLabel(
        content,
        text=(
            "⚠ For academic and research purposes only. "
            "This system is not a substitute for professional "
            "medical diagnosis."
        ),
        font=ctk.CTkFont(size=12),
        text_color=SECONDARY_TEXT
    ).pack(
        anchor="w",
        padx=30,
        pady=20
    )


def create_dashboard_card(
    parent,
    icon,
    title,
    description,
    accent,
    column
):

    card = ctk.CTkFrame(
        parent,
        fg_color=CARD_COLOR,
        corner_radius=15
    )

    card.grid(
        row=0,
        column=column,
        padx=8,
        sticky="nsew"
    )

    parent.grid_columnconfigure(
        column,
        weight=1
    )


    ctk.CTkLabel(
        card,
        text=icon,
        font=ctk.CTkFont(size=30)
    ).pack(
        anchor="w",
        padx=22,
        pady=(20, 5)
    )


    ctk.CTkLabel(
        card,
        text=title,
        font=ctk.CTkFont(
            size=19,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        anchor="w",
        padx=22
    )


    ctk.CTkLabel(
        card,
        text=description,
        font=ctk.CTkFont(size=13),
        text_color=SECONDARY_TEXT
    ).pack(
        anchor="w",
        padx=22,
        pady=(4, 20)
    )


def show_assessment():

    clear_content()

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ctk.CTkLabel(
        content,
        text="New Risk Assessment",
        font=ctk.CTkFont(
            size=30,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        anchor="w",
        padx=30,
        pady=(25, 3)
    )


    ctk.CTkLabel(
        content,
        text="Enter patient information and medical parameters",
        font=ctk.CTkFont(size=14),
        text_color=SECONDARY_TEXT
    ).pack(
        anchor="w",
        padx=30
    )


    # --------------------------------------------------------
    # SCROLLABLE AREA
    # --------------------------------------------------------

    scroll = ctk.CTkScrollableFrame(
        content,
        fg_color="transparent"
    )

    scroll.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=15
    )


    # --------------------------------------------------------
    # PATIENT INFORMATION CARD
    # --------------------------------------------------------

    patient_card = ctk.CTkFrame(
        scroll,
        fg_color=CARD_COLOR,
        corner_radius=15
    )

    patient_card.pack(
        fill="x",
        padx=10,
        pady=10
    )


    ctk.CTkLabel(
        patient_card,
        text="👤  Patient Information",
        font=ctk.CTkFont(
            size=20,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        anchor="w",
        padx=25,
        pady=(20, 15)
    )


    patient_grid = ctk.CTkFrame(
        patient_card,
        fg_color="transparent"
    )

    patient_grid.pack(
        fill="x",
        padx=20,
        pady=(0, 20)
    )


    global age_entry
    global gender_menu
    global bmi_entry
    global smoking_menu
    global alcohol_menu
    global activity_menu
    global family_menu


    age_entry = create_input(
        patient_grid,
        "Age",
        0,
        0
    )


    gender_menu = create_dropdown(
        patient_grid,
        "Gender",
        ["Male", "Female"],
        0,
        1
    )


    bmi_entry = create_input(
        patient_grid,
        "BMI",
        1,
        0
    )


    smoking_menu = create_dropdown(
        patient_grid,
        "Smoking Status",
        ["Never", "Former", "Current"],
        1,
        1
    )


    alcohol_menu = create_dropdown(
        patient_grid,
        "Alcohol Use",
        ["None", "Moderate", "Heavy"],
        2,
        0
    )


    activity_menu = create_dropdown(
        patient_grid,
        "Physical Activity",
        ["Low", "Moderate", "High"],
        2,
        1
    )


    family_menu = create_dropdown(
        patient_grid,
        "Family History",
        ["No", "Yes"],
        3,
        0
    )


    # --------------------------------------------------------
    # MEDICAL PARAMETERS
    # --------------------------------------------------------

    medical_card = ctk.CTkFrame(
        scroll,
        fg_color=CARD_COLOR,
        corner_radius=15
    )

    medical_card.pack(
        fill="x",
        padx=10,
        pady=10
    )


    ctk.CTkLabel(
        medical_card,
        text="🩺  Medical Parameters",
        font=ctk.CTkFont(
            size=20,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        anchor="w",
        padx=25,
        pady=(20, 15)
    )


    medical_grid = ctk.CTkFrame(
        medical_card,
        fg_color="transparent"
    )

    medical_grid.pack(
        fill="x",
        padx=20,
        pady=(0, 20)
    )


    global systolic_entry
    global diastolic_entry
    global heart_rate_entry
    global glucose_entry
    global cholesterol_entry
    global hdl_entry
    global ldl_entry
    global triglycerides_entry
    global creatinine_entry


    systolic_entry = create_input(
        medical_grid,
        "Systolic BP",
        0,
        0
    )


    diastolic_entry = create_input(
        medical_grid,
        "Diastolic BP",
        0,
        1
    )


    heart_rate_entry = create_input(
        medical_grid,
        "Heart Rate",
        1,
        0
    )


    glucose_entry = create_input(
        medical_grid,
        "Glucose (mg/dL)",
        1,
        1
    )


    cholesterol_entry = create_input(
        medical_grid,
        "Cholesterol (mg/dL)",
        2,
        0
    )


    hdl_entry = create_input(
        medical_grid,
        "HDL (mg/dL)",
        2,
        1
    )


    ldl_entry = create_input(
        medical_grid,
        "LDL (mg/dL)",
        3,
        0
    )


    triglycerides_entry = create_input(
        medical_grid,
        "Triglycerides (mg/dL)",
        3,
        1
    )


    creatinine_entry = create_input(
        medical_grid,
        "Creatinine (mg/dL)",
        4,
        0
    )


    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    button_frame = ctk.CTkFrame(
        scroll,
        fg_color="transparent"
    )

    button_frame.pack(
        fill="x",
        padx=10,
        pady=15
    )


    ctk.CTkButton(
        button_frame,
        text="🔍  ASSESS CHRONIC DISEASE RISK",
        height=52,
        font=ctk.CTkFont(
            size=16,
            weight="bold"
        ),
        command=predict_risk
    ).pack(
        side="left",
        fill="x",
        expand=True,
        padx=(0, 8)
    )


    ctk.CTkButton(
        button_frame,
        text="↻  Clear",
        height=52,
        width=150,
        fg_color="#E2E8F0",
        hover_color="#CBD5E1",
        text_color=TEXT_COLOR,
        command=clear_fields
    ).pack(
        side="right"
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    global diabetes_result
    global heart_result
    global kidney_result
    global overall_result
    global status_label


    result_card = ctk.CTkFrame(
        scroll,
        fg_color=CARD_COLOR,
        corner_radius=15
    )

    result_card.pack(
        fill="x",
        padx=10,
        pady=10
    )


    ctk.CTkLabel(
        result_card,
        text="📊  Risk Assessment Results",
        font=ctk.CTkFont(
            size=20,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        anchor="w",
        padx=25,
        pady=(20, 15)
    )


    results_frame = ctk.CTkFrame(
        result_card,
        fg_color="transparent"
    )

    results_frame.pack(
        fill="x",
        padx=20,
        pady=(0, 20)
    )


    diabetes_result = create_result_card(
        results_frame,
        "🩸  Diabetes",
        0
    )


    heart_result = create_result_card(
        results_frame,
        "❤️  Heart Disease",
        1
    )


    kidney_result = create_result_card(
        results_frame,
        "🫘  Kidney Disease",
        2
    )


    overall_result = ctk.CTkLabel(
        result_card,
        text="OVERALL RISK\n\nWaiting for assessment...",
        font=ctk.CTkFont(
            size=22,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    )

    overall_result.pack(
        pady=25
    )


    status_label = ctk.CTkLabel(
        result_card,
        text="",
        font=ctk.CTkFont(size=13),
        text_color=SECONDARY_TEXT
    )

    status_label.pack(
        pady=(0, 20)
    )


def create_input(
    parent,
    placeholder,
    row,
    column
):

    parent.grid_columnconfigure(
        column,
        weight=1
    )


    entry = ctk.CTkEntry(
        parent,
        placeholder_text=placeholder,
        height=42,
        corner_radius=8
    )

    entry.grid(
        row=row,
        column=column,
        padx=8,
        pady=7,
        sticky="ew"
    )

    return entry


def create_dropdown(
    parent,
    placeholder,
    values,
    row,
    column
):

    parent.grid_columnconfigure(
        column,
        weight=1
    )


    menu = ctk.CTkComboBox(
        parent,
        values=values,
        height=42,
        corner_radius=8
    )

    menu.set(
        values[0]
    )

    menu.grid(
        row=row,
        column=column,
        padx=8,
        pady=7,
        sticky="ew"
    )

    return menu


def create_result_card(
    parent,
    title,
    column
):

    parent.grid_columnconfigure(
        column,
        weight=1
    )


    frame = ctk.CTkFrame(
        parent,
        fg_color="#F8FAFC",
        corner_radius=12
    )

    frame.grid(
        row=0,
        column=column,
        padx=6,
        sticky="nsew"
    )


    label = ctk.CTkLabel(
        frame,
        text=title + "\n\nWaiting...",
        font=ctk.CTkFont(
            size=15,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    )

    label.pack(
        padx=15,
        pady=25
    )


    return label


def predict_risk():

    try:

        patient = {

            "age": float(
                age_entry.get()
            ),

            "gender":
                gender_menu.get(),

            "bmi": float(
                bmi_entry.get()
            ),

            "smoking_status":
                smoking_menu.get(),

            "alcohol_use":
                alcohol_menu.get(),

            "physical_activity":
                activity_menu.get(),

            "family_history_chronic_disease":
                family_menu.get(),

            "systolic_bp": float(
                systolic_entry.get()
            ),

            "diastolic_bp": float(
                diastolic_entry.get()
            ),

            "heart_rate": float(
                heart_rate_entry.get()
            ),

            "glucose_mg_dl": float(
                glucose_entry.get()
            ),

            "cholesterol_mg_dl": float(
                cholesterol_entry.get()
            ),

            "hdl_mg_dl": float(
                hdl_entry.get()
            ),

            "ldl_mg_dl": float(
                ldl_entry.get()
            ),

            "triglycerides_mg_dl": float(
                triglycerides_entry.get()
            ),

            "creatinine_mg_dl": float(
                creatinine_entry.get()
            )
        }


        patient_df = pd.DataFrame(
            [patient]
        )


        # ----------------------------------------------------
        # MODEL PREDICTIONS
        # ----------------------------------------------------

        diabetes_prediction = diabetes_model.predict(
            patient_df
        )[0]

        heart_prediction = heart_model.predict(
            patient_df
        )[0]

        kidney_prediction = kidney_model.predict(
            patient_df
        )[0]


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


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        diabetes_result.configure(
            text=(
                "🩸  DIABETES\n\n"
                f"{str(diabetes_prediction).upper()}\n"
                f"{diabetes_probability:.1f}%"
            )
        )


        heart_result.configure(
            text=(
                "❤️  HEART DISEASE\n\n"
                f"{str(heart_prediction).upper()}\n"
                f"{heart_probability:.1f}%"
            )
        )


        kidney_result.configure(
            text=(
                "🫘  KIDNEY DISEASE\n\n"
                f"{str(kidney_prediction).upper()}\n"
                f"{kidney_probability:.1f}%"
            )
        )


        risk_values = {
            "Low": 1,
            "Moderate": 2,
            "High": 3
        }


        highest_risk = max(

            risk_values.get(
                str(diabetes_prediction),
                2
            ),

            risk_values.get(
                str(heart_prediction),
                2
            ),

            risk_values.get(
                str(kidney_prediction),
                2
            )
        )


        if highest_risk == 3:

            overall = "HIGH RISK"

        elif highest_risk == 2:

            overall = "MODERATE RISK"

        else:

            overall = "LOW RISK"


        overall_result.configure(
            text=(
                "OVERALL RISK\n\n"
                f"{overall}"
            )
        )


        status_label.configure(
            text="✓ Assessment completed successfully.",
            text_color=GREEN
        )


    except ValueError:

        status_label.configure(
            text="⚠ Please enter valid numerical values.",
            text_color=RED
        )


    except Exception as e:

        status_label.configure(
            text=f"⚠ Error: {str(e)}",
            text_color=RED
        )


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
        text="🩸  DIABETES\n\nWaiting..."
    )


    heart_result.configure(
        text="❤️  HEART DISEASE\n\nWaiting..."
    )


    kidney_result.configure(
        text="🫘  KIDNEY DISEASE\n\nWaiting..."
    )


    overall_result.configure(
        text="OVERALL RISK\n\nWaiting for assessment..."
    )


    status_label.configure(
        text=""
    )


def clear_content():

    for widget in content.winfo_children():

        widget.destroy()



# ============================================================
# NEW UI MODULES
# ============================================================

HISTORY_FILE = os.path.join(BASE_DIR, "assessment_history.json")
_base_predict_risk = predict_risk


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def predict_risk():
    """Run the original working ML prediction and save a local history record."""
    _base_predict_risk()

    try:
        if status_label.cget("text").startswith("✓"):
            record = {
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "age": age_entry.get(),
                "gender": gender_menu.get(),
                "bmi": bmi_entry.get(),
                "smoking": smoking_menu.get(),
                "alcohol": alcohol_menu.get(),
                "activity": activity_menu.get(),
                "family_history": family_menu.get(),
                "systolic_bp": systolic_entry.get(),
                "diastolic_bp": diastolic_entry.get(),
                "heart_rate": heart_rate_entry.get(),
                "glucose": glucose_entry.get(),
                "cholesterol": cholesterol_entry.get(),
                "hdl": hdl_entry.get(),
                "ldl": ldl_entry.get(),
                "triglycerides": triglycerides_entry.get(),
                "creatinine": creatinine_entry.get(),
                "diabetes_result": diabetes_result.cget("text"),
                "heart_result": heart_result.cget("text"),
                "kidney_result": kidney_result.cget("text"),
                "overall_result": overall_result.cget("text")
            }
            history = load_history()
            history.append(record)
            save_history(history)
    except Exception:
        pass


def show_analytics():
    clear_content()
    history = load_history()

    ctk.CTkLabel(content, text="Analytics", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=30, pady=(25, 3))
    ctk.CTkLabel(content, text="Visual summary of completed machine-learning assessments", font=ctk.CTkFont(size=14), text_color=SECONDARY_TEXT).pack(anchor="w", padx=30)

    if not history:
        ctk.CTkLabel(content, text="No assessment data yet", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_COLOR).pack(pady=(150, 8))
        ctk.CTkLabel(content, text="Complete a Risk Assessment to generate analytics.", font=ctk.CTkFont(size=14), text_color=SECONDARY_TEXT).pack()
        return

    total = len(history)
    high = sum("HIGH" in x.get("overall_result", "").upper() for x in history)
    moderate = sum("MODERATE" in x.get("overall_result", "").upper() for x in history)
    low = sum("LOW" in x.get("overall_result", "").upper() for x in history)

    stats = ctk.CTkFrame(content, fg_color="transparent")
    stats.pack(fill="x", padx=25, pady=25)
    for i, (icon, title, value, color) in enumerate([
        ("📋", "Total Assessments", total, BLUE),
        ("🔴", "High Risk", high, RED),
        ("🟠", "Moderate Risk", moderate, ORANGE),
        ("🟢", "Low Risk", low, GREEN)
    ]):
        create_dashboard_card(stats, icon, title, f"{value} recorded", color, i)

    card = ctk.CTkFrame(content, fg_color=CARD_COLOR, corner_radius=15)
    card.pack(fill="x", padx=30, pady=10)
    ctk.CTkLabel(card, text="Overall Risk Distribution", font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=25, pady=(20, 15))

    for name, count, color in [("High Risk", high, RED), ("Moderate Risk", moderate, ORANGE), ("Low Risk", low, GREEN)]:
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=25, pady=8)
        ctk.CTkLabel(row, text=name, width=120, anchor="w", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_COLOR).pack(side="left")
        bar = ctk.CTkProgressBar(row, height=16, progress_color=color)
        bar.pack(side="left", fill="x", expand=True, padx=10)
        bar.set(count / total if total else 0)
        ctk.CTkLabel(row, text=f"{count}  ({count / total * 100:.1f}%)", width=120, anchor="e", text_color=SECONDARY_TEXT).pack(side="right")


def show_history():
    clear_content()
    history = load_history()

    ctk.CTkLabel(content, text="Assessment History", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=30, pady=(25, 3))
    ctk.CTkLabel(content, text="Previously completed risk assessments saved locally", font=ctk.CTkFont(size=14), text_color=SECONDARY_TEXT).pack(anchor="w", padx=30)

    toolbar = ctk.CTkFrame(content, fg_color="transparent")
    toolbar.pack(fill="x", padx=30, pady=18)

    search = ctk.CTkEntry(toolbar, placeholder_text="Search by date, risk or value...", width=360, height=40)
    search.pack(side="left")

    body = ctk.CTkScrollableFrame(content, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=20, pady=5)

    def render(*_):
        for w in body.winfo_children():
            w.destroy()
        q = search.get().lower().strip()
        filtered = [x for x in load_history() if not q or q in json.dumps(x).lower()]
        if not filtered:
            ctk.CTkLabel(body, text="No matching assessments found.", font=ctk.CTkFont(size=18, weight="bold"), text_color=SECONDARY_TEXT).pack(pady=60)
            return
        for item in reversed(filtered):
            card = ctk.CTkFrame(body, fg_color=CARD_COLOR, corner_radius=12)
            card.pack(fill="x", padx=10, pady=6)
            ctk.CTkLabel(card, text=item.get("timestamp", ""), font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=18, pady=(14, 5))
            ctk.CTkLabel(card, text=f"Diabetes: {item.get('diabetes_result', '').split(chr(10))[2] if len(item.get('diabetes_result','').split(chr(10))) > 2 else 'N/A'}    |    Heart: {item.get('heart_result', '').split(chr(10))[2] if len(item.get('heart_result','').split(chr(10))) > 2 else 'N/A'}    |    Kidney: {item.get('kidney_result', '').split(chr(10))[2] if len(item.get('kidney_result','').split(chr(10))) > 2 else 'N/A'}", font=ctk.CTkFont(size=12), text_color=SECONDARY_TEXT).pack(anchor="w", padx=18, pady=(0, 14))

    search.bind("<KeyRelease>", render)
    render()

    def clear_history():
        if messagebox.askyesno("Clear History", "Delete all saved assessment history?"):
            save_history([])
            render()

    ctk.CTkButton(toolbar, text="🗑 Clear History", width=130, height=40, fg_color=RED, hover_color="#B91C1C", command=clear_history).pack(side="right")


def build_report(item):
    return f"""EARLY MEDICAL DIAGNOSTICS ENGINE
CHRONIC DISEASE RISK ASSESSMENT REPORT
============================================================
Assessment Date : {item.get('timestamp', '')}

PATIENT / INPUT INFORMATION
Age             : {item.get('age', '')}
Gender          : {item.get('gender', '')}
BMI             : {item.get('bmi', '')}
Smoking         : {item.get('smoking', '')}
Alcohol         : {item.get('alcohol', '')}
Physical Activity: {item.get('activity', '')}
Family History  : {item.get('family_history', '')}

MEDICAL PARAMETERS
Systolic BP     : {item.get('systolic_bp', '')}
Diastolic BP    : {item.get('diastolic_bp', '')}
Heart Rate      : {item.get('heart_rate', '')}
Glucose         : {item.get('glucose', '')} mg/dL
Cholesterol     : {item.get('cholesterol', '')} mg/dL
HDL             : {item.get('hdl', '')} mg/dL
LDL             : {item.get('ldl', '')} mg/dL
Triglycerides   : {item.get('triglycerides', '')} mg/dL
Creatinine      : {item.get('creatinine', '')} mg/dL

MODEL RESULTS
------------------------------------------------------------
{item.get('diabetes_result', '')}

{item.get('heart_result', '')}

{item.get('kidney_result', '')}

{item.get('overall_result', '')}

============================================================
DISCLAIMER: Academic and research project only. This system is
not a substitute for professional medical diagnosis.
============================================================
"""


def export_report(item):
    path = filedialog.asksaveasfilename(
        title="Save Assessment Report",
        defaultextension=".txt",
        initialfile="medical_assessment_report.txt",
        filetypes=[("Text Report", "*.txt"), ("All Files", "*.*")]
    )
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_report(item))
        messagebox.showinfo("Report Exported", f"Report saved successfully:\n{path}")
    except Exception as e:
        messagebox.showerror("Report Error", str(e))


def show_reports():
    clear_content()
    history = load_history()

    ctk.CTkLabel(content, text="Reports", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=30, pady=(25, 3))
    ctk.CTkLabel(content, text="Export completed assessments as readable reports", font=ctk.CTkFont(size=14), text_color=SECONDARY_TEXT).pack(anchor="w", padx=30)

    if not history:
        ctk.CTkLabel(content, text="No reports available yet", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_COLOR).pack(pady=(150, 8))
        ctk.CTkLabel(content, text="Complete a risk assessment first.", text_color=SECONDARY_TEXT).pack()
        return

    body = ctk.CTkScrollableFrame(content, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=20, pady=20)

    for item in reversed(history):
        row = ctk.CTkFrame(body, fg_color=CARD_COLOR, corner_radius=12)
        row.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(row, text=f"Assessment — {item.get('timestamp', '')}", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_COLOR).pack(side="left", padx=18, pady=14)
        ctk.CTkButton(row, text="📄 Export", width=100, height=34, command=lambda x=item: export_report(x)).pack(side="right", padx=12)


def show_settings():
    clear_content()

    ctk.CTkLabel(content, text="Settings", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=30, pady=(25, 3))
    ctk.CTkLabel(content, text="Application configuration and model information", font=ctk.CTkFont(size=14), text_color=SECONDARY_TEXT).pack(anchor="w", padx=30)

    card = ctk.CTkFrame(content, fg_color=CARD_COLOR, corner_radius=15)
    card.pack(fill="x", padx=30, pady=25)

    ctk.CTkLabel(card, text="⚙  Appearance", font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=25, pady=(22, 10))
    mode = ctk.CTkComboBox(card, values=["Light", "Dark", "System"], width=220, height=40)
    mode.set("Light")
    mode.pack(anchor="w", padx=25, pady=(0, 22))
    mode.configure(command=lambda choice: ctk.set_appearance_mode(choice.lower()))

    model_card = ctk.CTkFrame(content, fg_color=CARD_COLOR, corner_radius=15)
    model_card.pack(fill="x", padx=30, pady=5)
    ctk.CTkLabel(model_card, text="🤖  Machine Learning Models", font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=25, pady=(22, 10))
    status = "LOADED ✓" if all(m is not None for m in [diabetes_model, heart_model, kidney_model]) else "ERROR ✗"
    ctk.CTkLabel(model_card, text=f"Status: {status}", font=ctk.CTkFont(size=15, weight="bold"), text_color=GREEN if status.startswith("LOADED") else RED).pack(anchor="w", padx=25, pady=4)
    ctk.CTkLabel(model_card, text="diabetes_model.pkl\nheart_model.pkl\nkidney_model.pkl", font=ctk.CTkFont(size=13), text_color=SECONDARY_TEXT, justify="left").pack(anchor="w", padx=25, pady=(0, 22))

    info = ctk.CTkFrame(content, fg_color=CARD_COLOR, corner_radius=15)
    info.pack(fill="x", padx=30, pady=12)
    ctk.CTkLabel(info, text="ℹ  Project Information", font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_COLOR).pack(anchor="w", padx=25, pady=(22, 10))
    ctk.CTkLabel(info, text="Early Medical Diagnostics Engine\nChronic Disease Risk Assessment\nPython + CustomTkinter + Scikit-learn\nModules: Diabetes, Heart Disease, Kidney Disease\nLocal storage: assessment_history.json", font=ctk.CTkFont(size=13), text_color=SECONDARY_TEXT, justify="left").pack(anchor="w", padx=25, pady=(0, 22))


# ============================================================
# SIDEBAR
# ============================================================

sidebar = ctk.CTkFrame(
    app,
    width=240,
    fg_color=SIDEBAR_COLOR,
    corner_radius=0
)

sidebar.pack(
    side="left",
    fill="y"
)

sidebar.pack_propagate(
    False
)


ctk.CTkLabel(
    sidebar,
    text="🏥",
    font=ctk.CTkFont(size=38)
).pack(
    pady=(30, 5)
)


ctk.CTkLabel(
    sidebar,
    text="EARLY MEDICAL",
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    ),
    text_color="white"
).pack()


ctk.CTkLabel(
    sidebar,
    text="DIAGNOSTICS ENGINE",
    font=ctk.CTkFont(
        size=12,
        weight="bold"
    ),
    text_color="#94A3B8"
).pack(
    pady=(0, 30)
)


def sidebar_button(
    text,
    command
):

    button = ctk.CTkButton(
        sidebar,
        text=text,
        height=45,
        corner_radius=8,
        fg_color="transparent",
        hover_color="#263449",
        text_color="#E2E8F0",
        anchor="w",
        font=ctk.CTkFont(
            size=14
        ),
        command=command
    )

    button.pack(
        fill="x",
        padx=15,
        pady=4
    )


sidebar_button(
    "🏠   Dashboard",
    show_dashboard
)

sidebar_button(
    "🔬   Risk Assessment",
    show_assessment
)


def show_coming_soon():

    clear_content()

    ctk.CTkLabel(
        content,
        text="Coming Soon",
        font=ctk.CTkFont(
            size=30,
            weight="bold"
        ),
        text_color=TEXT_COLOR
    ).pack(
        pady=(150, 10)
    )


    ctk.CTkLabel(
        content,
        text=(
            "This module will be added in the next "
            "development stage."
        ),
        font=ctk.CTkFont(size=15),
        text_color=SECONDARY_TEXT
    ).pack()


sidebar_button(
    "📊   Analytics",
    show_analytics
)

sidebar_button(
    "📋   Assessment History",
    show_history
)

sidebar_button(
    "📄   Reports",
    show_reports
)

sidebar_button(
    "⚙   Settings",
    show_settings
)


# ------------------------------------------------------------
# SIDEBAR FOOTER
# ------------------------------------------------------------

ctk.CTkLabel(
    sidebar,
    text=(
        "Machine Learning\n"
        "Risk Assessment System\n\n"
        "Academic Project"
    ),
    font=ctk.CTkFont(size=11),
    text_color="#64748B"
).pack(
    side="bottom",
    pady=25
)


# ============================================================
# CONTENT AREA
# ============================================================

content = ctk.CTkFrame(
    app,
    fg_color=BACKGROUND_COLOR,
    corner_radius=0
)

content.pack(
    side="left",
    fill="both",
    expand=True
)


# ============================================================
# START
# ============================================================

show_dashboard()

app.mainloop()