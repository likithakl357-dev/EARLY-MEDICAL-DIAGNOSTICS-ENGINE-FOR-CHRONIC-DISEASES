import customtkinter as ctk
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from tkinter import filedialog, messagebox


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# ============================================================
# COLORS
# ============================================================

SIDEBAR_COLOR = "#172033"
CARD_COLOR = "#FFFFFF"
BACKGROUND_COLOR = "#F4F7FB"

TEXT_COLOR = "#172033"
SECONDARY_TEXT = "#64748B"

BLUE_COLOR = "#2563EB"
GREEN_COLOR = "#16A34A"
ORANGE_COLOR = "#F59E0B"
RED_COLOR = "#DC2626"

LIGHT_BLUE = "#EFF6FF"
LIGHT_GREEN = "#F0FDF4"
LIGHT_ORANGE = "#FFFBEB"
LIGHT_RED = "#FEF2F2"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "medical_models"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "assessment_history.json"
)

DIABETES_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "diabetes_model.pkl"
)

HEART_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "heart_model.pkl"
)

KIDNEY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "kidney_model.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

diabetes_model = None
heart_model = None
kidney_model = None

model_errors = []


def load_models():

    global diabetes_model
    global heart_model
    global kidney_model

    model_errors.clear()

    try:
        diabetes_model = joblib.load(
            DIABETES_MODEL_PATH
        )
    except Exception as e:
        diabetes_model = None
        model_errors.append(
            f"Diabetes model: {str(e)}"
        )

    try:
        heart_model = joblib.load(
            HEART_MODEL_PATH
        )
    except Exception as e:
        heart_model = None
        model_errors.append(
            f"Heart model: {str(e)}"
        )

    try:
        kidney_model = joblib.load(
            KIDNEY_MODEL_PATH
        )
    except Exception as e:
        kidney_model = None
        model_errors.append(
            f"Kidney model: {str(e)}"
        )


load_models()


# ============================================================
# MAIN APPLICATION
# ============================================================

class MedicalDiagnosticsApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(
            "Early Medical Diagnostics Engine"
        )

        self.geometry(
            "1400x850"
        )

        self.minsize(
            1200,
            750
        )

        self.configure(
            fg_color=BACKGROUND_COLOR
        )

        # Store current page
        self.current_page = None

        # Store result widgets
        self.result_cards = {}

        # Store assessment fields
        self.fields = {}

        # Build sidebar
        self.create_sidebar()

        # Show dashboard
        self.show_dashboard()


    # ========================================================
    # SIDEBAR
    # ========================================================

    def create_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self,
            width=240,
            corner_radius=0,
            fg_color=SIDEBAR_COLOR
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        self.sidebar.pack_propagate(False)


        # ----------------------------------------------------
        # Application title
        # ----------------------------------------------------

        title_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        title_frame.pack(
            fill="x",
            padx=20,
            pady=(30, 20)
        )

        title_label = ctk.CTkLabel(
            title_frame,
            text="MEDICAL\nDIAGNOSTICS",
            font=ctk.CTkFont(
                size=20,
                weight="bold"
            ),
            text_color="white",
            justify="left"
        )

        title_label.pack(
            anchor="w"
        )


        subtitle = ctk.CTkLabel(
            title_frame,
            text="Early Risk Assessment",
            font=ctk.CTkFont(
                size=11
            ),
            text_color="#CBD5E1"
        )

        subtitle.pack(
            anchor="w",
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        self.create_nav_button(
            "Dashboard",
            self.show_dashboard
        )

        self.create_nav_button(
            "Risk Assessment",
            self.show_assessment
        )

        self.create_nav_button(
            "Analytics",
            self.show_analytics
        )

        self.create_nav_button(
            "Assessment History",
            self.show_history
        )

        self.create_nav_button(
            "Reports",
            self.show_reports
        )

        self.create_nav_button(
            "Settings",
            self.show_settings
        )


        # ----------------------------------------------------
        # Bottom information
        # ----------------------------------------------------

        bottom_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        bottom_frame.pack(
            side="bottom",
            fill="x",
            padx=20,
            pady=20
        )

        version_label = ctk.CTkLabel(
            bottom_frame,
            text="Academic Project\nVersion 1.0",
            font=ctk.CTkFont(
                size=10
            ),
            text_color="#94A3B8",
            justify="left"
        )

        version_label.pack(
            anchor="w"
        )


    def create_nav_button(
        self,
        text,
        command
    ):

        button = ctk.CTkButton(
            self.sidebar,
            text=text,
            command=command,
            height=45,
            corner_radius=8,
            fg_color="transparent",
            hover_color="#26344F",
            text_color="#E2E8F0",
            anchor="w",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        button.pack(
            fill="x",
            padx=15,
            pady=4
        )


    # ========================================================
    # PAGE MANAGEMENT
    # ========================================================

    def clear_page(self):

        if self.current_page is not None:

            self.current_page.destroy()

            self.current_page = None


    def create_page(self):

        self.clear_page()

        self.current_page = ctk.CTkFrame(
            self,
            fg_color=BACKGROUND_COLOR,
            corner_radius=0
        )

        self.current_page.pack(
            side="right",
            fill="both",
            expand=True
        )

        return self.current_page


    # ========================================================
    # DASHBOARD
    # ========================================================

    def show_dashboard(self):

        page = self.create_page()


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )

        header.pack(
            fill="x",
            padx=35,
            pady=(30, 10)
        )


        heading = ctk.CTkLabel(
            header,
            text="Medical Diagnostics Dashboard",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        )

        heading.pack(
            anchor="w"
        )


        subheading = ctk.CTkLabel(
            header,
            text="Early risk assessment for chronic diseases using machine learning",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT
        )

        subheading.pack(
            anchor="w",
            pady=(5, 0)
        )


        # ----------------------------------------------------
        # Disease cards
        # ----------------------------------------------------

        cards_frame = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )

        cards_frame.pack(
            fill="x",
            padx=35,
            pady=20
        )

        cards_frame.grid_columnconfigure(
            (0, 1, 2),
            weight=1
        )


        self.create_dashboard_card(
            cards_frame,
            "Diabetes Risk",
            "Predict diabetes risk level",
            "DIABETES",
            0
        )

        self.create_dashboard_card(
            cards_frame,
            "Heart Disease Risk",
            "Predict cardiovascular risk level",
            "HEART",
            1
        )

        self.create_dashboard_card(
            cards_frame,
            "Kidney Disease Risk",
            "Predict kidney disease risk level",
            "KIDNEY",
            2
        )


        # ----------------------------------------------------
        # Welcome card
        # ----------------------------------------------------

        welcome = ctk.CTkFrame(
            page,
            fg_color=CARD_COLOR,
            corner_radius=16
        )

        welcome.pack(
            fill="x",
            padx=35,
            pady=10
        )


        welcome_title = ctk.CTkLabel(
            welcome,
            text="Welcome to the Early Medical Diagnostics Engine",
            font=ctk.CTkFont(
                size=20,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        )

        welcome_title.pack(
            anchor="w",
            padx=25,
            pady=(25, 8)
        )


        welcome_text = ctk.CTkLabel(
            welcome,
            text=(
                "Enter patient information and medical parameters to "
                "estimate risk levels for diabetes, heart disease, "
                "and kidney disease."
            ),
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT,
            justify="left"
        )

        welcome_text.pack(
            anchor="w",
            padx=25,
            pady=(0, 18)
        )


        start_button = ctk.CTkButton(
            welcome,
            text="Start Risk Assessment",
            command=self.show_assessment,
            width=220,
            height=45,
            corner_radius=10,
            fg_color=BLUE_COLOR,
            hover_color="#1D4ED8",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        start_button.pack(
            anchor="w",
            padx=25,
            pady=(0, 25)
        )


        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        disclaimer = ctk.CTkLabel(
            page,
            text=(
                "⚠ Academic / Research Use Only — "
                "This system provides machine-learning risk estimates "
                "and is not a substitute for professional medical diagnosis."
            ),
            font=ctk.CTkFont(
                size=11
            ),
            text_color="#92400E",
            fg_color=LIGHT_ORANGE,
            corner_radius=8
        )

        disclaimer.pack(
            fill="x",
            padx=35,
            pady=(10, 20),
            ipady=10
        )


    def create_dashboard_card(
        self,
        parent,
        title,
        description,
        disease,
        column
    ):

        card = ctk.CTkFrame(
            parent,
            fg_color=CARD_COLOR,
            corner_radius=16
        )

        card.grid(
            row=0,
            column=column,
            padx=8,
            pady=5,
            sticky="nsew"
        )


        icon = ctk.CTkLabel(
            card,
            text="●",
            font=ctk.CTkFont(
                size=30,
                weight="bold"
            ),
            text_color=BLUE_COLOR
        )

        icon.pack(
            anchor="w",
            padx=20,
            pady=(20, 5)
        )


        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(
                size=17,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        )

        title_label.pack(
            anchor="w",
            padx=20
        )


        desc = ctk.CTkLabel(
            card,
            text=description,
            font=ctk.CTkFont(
                size=11
            ),
            text_color=SECONDARY_TEXT
        )

        desc.pack(
            anchor="w",
            padx=20,
            pady=(4, 20)
        )


        status = ctk.CTkLabel(
            card,
            text="Model Ready" if self.get_model(disease) else "Model Error",
            font=ctk.CTkFont(
                size=11,
                weight="bold"
            ),
            text_color=GREEN_COLOR if self.get_model(disease) else RED_COLOR
        )

        status.pack(
            anchor="w",
            padx=20,
            pady=(0, 20)
        )


    # ========================================================
    # MODEL HELPER
    # ========================================================

    def get_model(self, disease):

        if disease == "DIABETES":
            return diabetes_model

        if disease == "HEART":
            return heart_model

        if disease == "KIDNEY":
            return kidney_model

        return None


    # ========================================================
    # ASSESSMENT PAGE
    # ========================================================

    def show_assessment(self):

        page = self.create_page()


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )

        header.pack(
            fill="x",
            padx=30,
            pady=(25, 5)
        )


        heading = ctk.CTkLabel(
            header,
            text="Risk Assessment",
            font=ctk.CTkFont(
                size=27,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        )

        heading.pack(
            anchor="w"
        )


        subheading = ctk.CTkLabel(
            header,
            text="Enter patient details and clinical measurements",
            font=ctk.CTkFont(
                size=12
            ),
            text_color=SECONDARY_TEXT
        )

        subheading.pack(
            anchor="w",
            pady=(4, 0)
        )


        # ----------------------------------------------------
        # Scrollable content
        # ----------------------------------------------------

        scroll = ctk.CTkScrollableFrame(
            page,
            fg_color="transparent"
        )

        scroll.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=10
        )


        # ----------------------------------------------------
        # Patient information card
        # ----------------------------------------------------

        patient_card = ctk.CTkFrame(
            scroll,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        patient_card.pack(
            fill="x",
            pady=8
        )


        ctk.CTkLabel(
            patient_card,
            text="Patient Information",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=20,
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


        patient_grid.grid_columnconfigure(
            (0, 1),
            weight=1
        )


        self.create_entry_field(
            patient_grid,
            "Age",
            "age",
            0,
            0
        )

        self.create_combo_field(
            patient_grid,
            "Gender",
            "gender",
            ["Male", "Female"],
            0,
            1
        )

        self.create_entry_field(
            patient_grid,
            "BMI",
            "bmi",
            1,
            0
        )

        self.create_combo_field(
            patient_grid,
            "Smoking Status",
            "smoking_status",
            ["Never", "Former", "Current"],
            1,
            1
        )

        self.create_combo_field(
            patient_grid,
            "Alcohol Use",
            "alcohol_use",
            ["None", "Moderate", "Heavy"],
            2,
            0
        )

        self.create_combo_field(
            patient_grid,
            "Physical Activity",
            "physical_activity",
            ["Low", "Moderate", "High"],
            2,
            1
        )

        self.create_combo_field(
            patient_grid,
            "Family History of Chronic Disease",
            "family_history_chronic_disease",
            ["No", "Yes"],
            3,
            0
        )


        # ----------------------------------------------------
        # Medical parameters
        # ----------------------------------------------------

        medical_card = ctk.CTkFrame(
            scroll,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        medical_card.pack(
            fill="x",
            pady=8
        )


        ctk.CTkLabel(
            medical_card,
            text="Medical Parameters",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=20,
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


        medical_grid.grid_columnconfigure(
            (0, 1, 2),
            weight=1
        )


        medical_fields = [

            ("Systolic BP", "systolic_bp", 0, 0),

            ("Diastolic BP", "diastolic_bp", 0, 1),

            ("Heart Rate", "heart_rate", 0, 2),

            ("Glucose (mg/dL)", "glucose_mg_dl", 1, 0),

            ("Cholesterol (mg/dL)", "cholesterol_mg_dl", 1, 1),

            ("HDL (mg/dL)", "hdl_mg_dl", 1, 2),

            ("LDL (mg/dL)", "ldl_mg_dl", 2, 0),

            ("Triglycerides (mg/dL)", "triglycerides_mg_dl", 2, 1),

            ("Creatinine (mg/dL)", "creatinine_mg_dl", 2, 2)

        ]


        for label, key, row, column in medical_fields:

            self.create_entry_field(
                medical_grid,
                label,
                key,
                row,
                column
            )


        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = ctk.CTkFrame(
            scroll,
            fg_color="transparent"
        )

        button_frame.pack(
            fill="x",
            pady=15
        )


        predict_button = ctk.CTkButton(
            button_frame,
            text="Calculate Risk",
            command=self.predict_risk,
            width=200,
            height=45,
            corner_radius=10,
            fg_color=BLUE_COLOR,
            hover_color="#1D4ED8",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        predict_button.pack(
            side="left",
            padx=(0, 10)
        )


        clear_button = ctk.CTkButton(
            button_frame,
            text="Clear",
            command=self.clear_fields,
            width=130,
            height=45,
            corner_radius=10,
            fg_color="#E2E8F0",
            hover_color="#CBD5E1",
            text_color=TEXT_COLOR,
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        clear_button.pack(
            side="left"
        )


        # ----------------------------------------------------
        # Result card
        # ----------------------------------------------------

        result_section = ctk.CTkFrame(
            scroll,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        result_section.pack(
            fill="x",
            pady=(8, 20)
        )


        ctk.CTkLabel(
            result_section,
            text="Risk Prediction Results",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=20,
            pady=(20, 10)
        )


        result_grid = ctk.CTkFrame(
            result_section,
            fg_color="transparent"
        )

        result_grid.pack(
            fill="x",
            padx=15,
            pady=(0, 15)
        )


        result_grid.grid_columnconfigure(
            (0, 1, 2),
            weight=1
        )


        self.result_cards = {}

        self.result_cards["diabetes"] = self.create_result_card(
            result_grid,
            "Diabetes Risk",
            0
        )

        self.result_cards["heart"] = self.create_result_card(
            result_grid,
            "Heart Disease Risk",
            1
        )

        self.result_cards["kidney"] = self.create_result_card(
            result_grid,
            "Kidney Disease Risk",
            2
        )


        # Overall result

        self.overall_frame = ctk.CTkFrame(
            result_section,
            fg_color=LIGHT_BLUE,
            corner_radius=12
        )

        self.overall_frame.pack(
            fill="x",
            padx=20,
            pady=(5, 20)
        )


        self.overall_label = ctk.CTkLabel(
            self.overall_frame,
            text="Overall Risk: WAITING",
            font=ctk.CTkFont(
                size=20,
                weight="bold"
            ),
            text_color=SECONDARY_TEXT
        )

        self.overall_label.pack(
            pady=18
        )


    # ========================================================
    # FIELD CREATION
    # ========================================================

    def create_entry_field(
        self,
        parent,
        label,
        key,
        row,
        column
    ):

        frame = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        frame.grid(
            row=row,
            column=column,
            padx=8,
            pady=8,
            sticky="ew"
        )


        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            pady=(0, 5)
        )


        entry = ctk.CTkEntry(
            frame,
            height=38,
            corner_radius=8,
            border_width=1,
            border_color="#CBD5E1",
            fg_color="#FFFFFF",
            text_color=TEXT_COLOR
        )

        entry.pack(
            fill="x"
        )


        self.fields[key] = entry


    def create_combo_field(
        self,
        parent,
        label,
        key,
        values,
        row,
        column
    ):

        frame = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        frame.grid(
            row=row,
            column=column,
            padx=8,
            pady=8,
            sticky="ew"
        )


        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            pady=(0, 5)
        )


        combo = ctk.CTkComboBox(
            frame,
            values=values,
            height=38,
            corner_radius=8,
            border_width=1,
            border_color="#CBD5E1",
            fg_color="#FFFFFF",
            text_color=TEXT_COLOR,
            button_color=BLUE_COLOR
        )

        combo.set(
            values[0]
        )

        combo.pack(
            fill="x"
        )


        self.fields[key] = combo


    # ========================================================
    # RESULT CARD
    # ========================================================

    def create_result_card(
        self,
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
            corner_radius=14
        )

        frame.grid(
            row=0,
            column=column,
            padx=8,
            pady=5,
            sticky="nsew"
        )


        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        )

        title_label.pack(
            pady=(18, 8)
        )


        risk_label = ctk.CTkLabel(
            frame,
            text="WAITING",
            font=ctk.CTkFont(
                size=22,
                weight="bold"
            ),
            text_color=SECONDARY_TEXT
        )

        risk_label.pack(
            pady=(2, 5)
        )


        probability_label = ctk.CTkLabel(
            frame,
            text="Probability: --",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT
        )

        probability_label.pack(
            pady=(0, 10)
        )


        probability_bar = ctk.CTkProgressBar(
            frame,
            width=180,
            height=12,
            corner_radius=6
        )

        probability_bar.pack(
            pady=(0, 20)
        )

        probability_bar.set(
            0
        )


        result_widgets = {
            "frame": frame,
            "risk": risk_label,
            "probability": probability_label,
            "bar": probability_bar
        }


        return result_widgets


    # ========================================================
    # GET FIELD VALUE
    # ========================================================

    def get_field_value(self, key):

        widget = self.fields.get(key)

        if widget is None:
            return ""

        return widget.get().strip()


    # ========================================================
    # PREDICTED CLASS PROBABILITY
    # ========================================================

    def get_model_prediction(
        self,
        model,
        patient_df
    ):

        prediction = model.predict(
            patient_df
        )[0]

        probability = 0.0


        try:

            probabilities = model.predict_proba(
                patient_df
            )[0]

            classes = list(
                model.classes_
            )

            if prediction in classes:

                index = classes.index(
                    prediction
                )

                probability = float(
                    probabilities[index]
                ) * 100

            else:

                probability = float(
                    max(probabilities)
                ) * 100

        except Exception:

            probability = 0.0


        return (
            str(prediction),
            probability
        )


    # ========================================================
    # PREDICT RISK
    # ========================================================

    def predict_risk(self):

        # ----------------------------------------------------
        # Check models
        # ----------------------------------------------------

        if (
            diabetes_model is None
            or heart_model is None
            or kidney_model is None
        ):

            messagebox.showerror(
                "Model Error",
                "One or more machine learning models could not be loaded.\n\n"
                + "\n".join(model_errors)
            )

            return


        # ----------------------------------------------------
        # Validate numeric fields
        # ----------------------------------------------------

        numeric_fields = [

            "age",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "glucose_mg_dl",
            "cholesterol_mg_dl",
            "hdl_mg_dl",
            "ldl_mg_dl",
            "triglycerides_mg_dl",
            "creatinine_mg_dl"

        ]


        numeric_values = {}


        for key in numeric_fields:

            value = self.get_field_value(
                key
            )

            if value == "":

                messagebox.showwarning(
                    "Missing Information",
                    f"Please enter {key.replace('_', ' ').title()}."
                )

                return


            try:

                numeric_values[key] = float(
                    value
                )

            except ValueError:

                messagebox.showwarning(
                    "Invalid Value",
                    f"Please enter a valid number for {key.replace('_', ' ').title()}."
                )

                return


        # ----------------------------------------------------
        # Read categorical fields
        # ----------------------------------------------------

        gender = self.get_field_value(
            "gender"
        )

        smoking_status = self.get_field_value(
            "smoking_status"
        )

        alcohol_use = self.get_field_value(
            "alcohol_use"
        )

        physical_activity = self.get_field_value(
            "physical_activity"
        )

        family_history = self.get_field_value(
            "family_history_chronic_disease"
        )


        # ----------------------------------------------------
        # Create patient dataframe
        # ----------------------------------------------------

        patient_data = {

            "age":
                numeric_values["age"],

            "gender":
                gender,

            "bmi":
                numeric_values["bmi"],

            "smoking_status":
                smoking_status,

            "alcohol_use":
                alcohol_use,

            "physical_activity":
                physical_activity,

            "family_history_chronic_disease":
                family_history,

            "systolic_bp":
                numeric_values["systolic_bp"],

            "diastolic_bp":
                numeric_values["diastolic_bp"],

            "heart_rate":
                numeric_values["heart_rate"],

            "glucose_mg_dl":
                numeric_values["glucose_mg_dl"],

            "cholesterol_mg_dl":
                numeric_values["cholesterol_mg_dl"],

            "hdl_mg_dl":
                numeric_values["hdl_mg_dl"],

            "ldl_mg_dl":
                numeric_values["ldl_mg_dl"],

            "triglycerides_mg_dl":
                numeric_values["triglycerides_mg_dl"],

            "creatinine_mg_dl":
                numeric_values["creatinine_mg_dl"]

        }


        patient_df = pd.DataFrame(
            [patient_data]
        )


        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        try:

            diabetes_risk, diabetes_probability = (
                self.get_model_prediction(
                    diabetes_model,
                    patient_df
                )
            )


            heart_risk, heart_probability = (
                self.get_model_prediction(
                    heart_model,
                    patient_df
                )
            )


            kidney_risk, kidney_probability = (
                self.get_model_prediction(
                    kidney_model,
                    patient_df
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Prediction Error",
                "An error occurred while generating the prediction:\n\n"
                + str(e)
            )

            return


        # ----------------------------------------------------
        # Update result cards
        # ----------------------------------------------------

        self.update_result_card(
            self.result_cards["diabetes"],
            diabetes_risk,
            diabetes_probability
        )


        self.update_result_card(
            self.result_cards["heart"],
            heart_risk,
            heart_probability
        )


        self.update_result_card(
            self.result_cards["kidney"],
            kidney_risk,
            kidney_probability
        )


        # ----------------------------------------------------
        # Overall risk
        # ----------------------------------------------------

        risk_values = {

            "Low": 1,
            "Moderate": 2,
            "High": 3

        }


        overall_score = max(

            risk_values.get(
                diabetes_risk,
                1
            ),

            risk_values.get(
                heart_risk,
                1
            ),

            risk_values.get(
                kidney_risk,
                1
            )

        )


        if overall_score == 3:

            overall_risk = "HIGH RISK"

            overall_color = RED_COLOR

            overall_bg = LIGHT_RED


        elif overall_score == 2:

            overall_risk = "MODERATE RISK"

            overall_color = ORANGE_COLOR

            overall_bg = LIGHT_ORANGE


        else:

            overall_risk = "LOW RISK"

            overall_color = GREEN_COLOR

            overall_bg = LIGHT_GREEN


        self.overall_frame.configure(
            fg_color=overall_bg
        )


        self.overall_label.configure(
            text=f"Overall Risk: {overall_risk}",
            text_color=overall_color
        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history_record = {

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "patient":

                patient_data,

            "predictions": {

                "diabetes": {

                    "risk":
                        diabetes_risk,

                    "probability":
                        round(
                            diabetes_probability,
                            2
                        )

                },

                "heart": {

                    "risk":
                        heart_risk,

                    "probability":
                        round(
                            heart_probability,
                            2
                        )

                },

                "kidney": {

                    "risk":
                        kidney_risk,

                    "probability":
                        round(
                            kidney_probability,
                            2
                        )

                },

                "overall":
                    overall_risk

            }

        }


        self.save_history(
            history_record
        )


        messagebox.showinfo(
            "Assessment Complete",
            "Risk assessment completed successfully."
        )


    # ========================================================
    # UPDATE RESULT CARD
    # ========================================================

    def update_result_card(
        self,
        widgets,
        risk,
        probability
    ):

        risk_upper = risk.upper()


        if risk_upper == "HIGH":

            text_color = RED_COLOR

            background = LIGHT_RED

        elif risk_upper == "MODERATE":

            text_color = ORANGE_COLOR

            background = LIGHT_ORANGE

        else:

            text_color = GREEN_COLOR

            background = LIGHT_GREEN


        widgets["frame"].configure(
            fg_color=background
        )


        widgets["risk"].configure(
            text=risk_upper,
            text_color=text_color
        )


        widgets["probability"].configure(
            text=f"Probability: {probability:.1f}%",
            text_color=SECONDARY_TEXT
        )


        widgets["bar"].set(
            max(
                0,
                min(
                    probability / 100,
                    1
                )
            )
        )


    # ========================================================
    # CLEAR FIELDS
    # ========================================================

    def clear_fields(self):

        for key, widget in self.fields.items():

            if isinstance(
                widget,
                ctk.CTkEntry
            ):

                widget.delete(
                    0,
                    "end"
                )

            elif isinstance(
                widget,
                ctk.CTkComboBox
            ):

                if key == "gender":

                    widget.set(
                        "Male"
                    )

                elif key == "smoking_status":

                    widget.set(
                        "Never"
                    )

                elif key == "alcohol_use":

                    widget.set(
                        "None"
                    )

                elif key == "physical_activity":

                    widget.set(
                        "Low"
                    )

                elif key == "family_history_chronic_disease":

                    widget.set(
                        "No"
                    )


        # Reset result cards

        for widgets in self.result_cards.values():

            widgets["frame"].configure(
                fg_color="#F8FAFC"
            )

            widgets["risk"].configure(
                text="WAITING",
                text_color=SECONDARY_TEXT
            )

            widgets["probability"].configure(
                text="Probability: --",
                text_color=SECONDARY_TEXT
            )

            widgets["bar"].set(
                0
            )


        if hasattr(
            self,
            "overall_frame"
        ):

            self.overall_frame.configure(
                fg_color=LIGHT_BLUE
            )


        if hasattr(
            self,
            "overall_label"
        ):

            self.overall_label.configure(
                text="Overall Risk: WAITING",
                text_color=SECONDARY_TEXT
            )


    # ========================================================
    # HISTORY
    # ========================================================

    def save_history(
        self,
        record
    ):

        history = []


        if os.path.exists(
            HISTORY_FILE
        ):

            try:

                with open(
                    HISTORY_FILE,
                    "r",
                    encoding="utf-8"
                ) as file:

                    history = json.load(
                        file
                    )

            except Exception:

                history = []


        history.append(
            record
        )


        try:

            with open(
                HISTORY_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    history,
                    file,
                    indent=4
                )

        except Exception as e:

            print(
                "Could not save history:",
                e
            )


    def load_history(self):

        if not os.path.exists(
            HISTORY_FILE
        ):

            return []


        try:

            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(
                    file
                )

        except Exception:

            return []


    # ========================================================
    # ANALYTICS
    # ========================================================

    def show_analytics(self):

        page = self.create_page()


        history = self.load_history()


        # Header

        ctk.CTkLabel(
            page,
            text="Analytics",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=35,
            pady=(30, 5)
        )


        ctk.CTkLabel(
            page,
            text="Summary of completed risk assessments",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT
        ).pack(
            anchor="w",
            padx=35
        )


        # Statistics

        stats_frame = ctk.CTkFrame(
            page,
            fg_color="transparent"
        )

        stats_frame.pack(
            fill="x",
            padx=35,
            pady=25
        )


        stats_frame.grid_columnconfigure(
            (0, 1, 2, 3),
            weight=1
        )


        total = len(
            history
        )

        high = 0
        moderate = 0
        low = 0


        for record in history:

            overall = record.get(
                "predictions",
                {}
            ).get(
                "overall",
                ""
            )


            if "HIGH" in overall:

                high += 1

            elif "MODERATE" in overall:

                moderate += 1

            elif "LOW" in overall:

                low += 1


        self.create_stat_card(
            stats_frame,
            "Total Assessments",
            str(total),
            BLUE_COLOR,
            0
        )


        self.create_stat_card(
            stats_frame,
            "High Risk",
            str(high),
            RED_COLOR,
            1
        )


        self.create_stat_card(
            stats_frame,
            "Moderate Risk",
            str(moderate),
            ORANGE_COLOR,
            2
        )


        self.create_stat_card(
            stats_frame,
            "Low Risk",
            str(low),
            GREEN_COLOR,
            3
        )


        # Distribution

        distribution = ctk.CTkFrame(
            page,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        distribution.pack(
            fill="both",
            expand=True,
            padx=35,
            pady=10
        )


        ctk.CTkLabel(
            distribution,
            text="Risk Distribution",
            font=ctk.CTkFont(
                size=19,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=25,
            pady=(25, 15)
        )


        self.create_distribution_row(
            distribution,
            "High Risk",
            high,
            total,
            RED_COLOR
        )


        self.create_distribution_row(
            distribution,
            "Moderate Risk",
            moderate,
            total,
            ORANGE_COLOR
        )


        self.create_distribution_row(
            distribution,
            "Low Risk",
            low,
            total,
            GREEN_COLOR
        )


    def create_stat_card(
        self,
        parent,
        title,
        value,
        color,
        column
    ):

        card = ctk.CTkFrame(
            parent,
            fg_color=CARD_COLOR,
            corner_radius=14
        )

        card.grid(
            row=0,
            column=column,
            padx=7,
            sticky="nsew"
        )


        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(
                size=12
            ),
            text_color=SECONDARY_TEXT
        ).pack(
            pady=(20, 5)
        )


        ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            ),
            text_color=color
        ).pack(
            pady=(0, 20)
        )


    def create_distribution_row(
        self,
        parent,
        title,
        count,
        total,
        color
    ):

        row = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        row.pack(
            fill="x",
            padx=25,
            pady=12
        )


        percentage = (
            count / total * 100
            if total > 0
            else 0
        )


        label = ctk.CTkLabel(
            row,
            text=f"{title}: {count}",
            width=160,
            anchor="w",
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        )

        label.pack(
            side="left"
        )


        bar = ctk.CTkProgressBar(
            row,
            height=12
        )

        bar.pack(
            side="left",
            fill="x",
            expand=True,
            padx=10
        )

        bar.set(
            percentage / 100
        )


        ctk.CTkLabel(
            row,
            text=f"{percentage:.1f}%",
            width=70,
            text_color=SECONDARY_TEXT
        ).pack(
            side="right"
        )


    # ========================================================
    # HISTORY PAGE
    # ========================================================

    def show_history(self):

        page = self.create_page()


        ctk.CTkLabel(
            page,
            text="Assessment History",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=35,
            pady=(30, 5)
        )


        ctk.CTkLabel(
            page,
            text="Previous risk assessments stored on this computer",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT
        ).pack(
            anchor="w",
            padx=35
        )


        history = self.load_history()


        if not history:

            empty = ctk.CTkFrame(
                page,
                fg_color=CARD_COLOR,
                corner_radius=15
            )

            empty.pack(
                fill="both",
                expand=True,
                padx=35,
                pady=30
            )


            ctk.CTkLabel(
                empty,
                text="No assessment history yet.",
                font=ctk.CTkFont(
                    size=18,
                    weight="bold"
                ),
                text_color=SECONDARY_TEXT
            ).pack(
                pady=80
            )

            return


        scroll = ctk.CTkScrollableFrame(
            page,
            fg_color="transparent"
        )

        scroll.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20
        )


        # Display newest first

        for index, record in enumerate(
            reversed(history),
            start=1
        ):

            predictions = record.get(
                "predictions",
                {}
            )


            card = ctk.CTkFrame(
                scroll,
                fg_color=CARD_COLOR,
                corner_radius=12
            )

            card.pack(
                fill="x",
                pady=7
            )


            timestamp = record.get(
                "timestamp",
                "Unknown"
            )


            overall = predictions.get(
                "overall",
                "Unknown"
            )


            ctk.CTkLabel(
                card,
                text=f"Assessment {index}",
                font=ctk.CTkFont(
                    size=16,
                    weight="bold"
                ),
                text_color=TEXT_COLOR
            ).pack(
                anchor="w",
                padx=20,
                pady=(15, 3)
            )


            ctk.CTkLabel(
                card,
                text=f"Date: {timestamp}",
                font=ctk.CTkFont(
                    size=11
                ),
                text_color=SECONDARY_TEXT
            ).pack(
                anchor="w",
                padx=20
            )


            ctk.CTkLabel(
                card,
                text=f"Overall Result: {overall}",
                font=ctk.CTkFont(
                    size=13,
                    weight="bold"
                ),
                text_color=(
                    RED_COLOR
                    if "HIGH" in overall
                    else ORANGE_COLOR
                    if "MODERATE" in overall
                    else GREEN_COLOR
                )
            ).pack(
                anchor="w",
                padx=20,
                pady=(8, 15)
            )


    # ========================================================
    # REPORTS PAGE
    # ========================================================

    def show_reports(self):

        page = self.create_page()


        ctk.CTkLabel(
            page,
            text="Reports",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=35,
            pady=(30, 5)
        )


        ctk.CTkLabel(
            page,
            text="Export assessment history for documentation",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT
        ).pack(
            anchor="w",
            padx=35
        )


        card = ctk.CTkFrame(
            page,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        card.pack(
            fill="x",
            padx=35,
            pady=30
        )


        ctk.CTkLabel(
            card,
            text="Generate Report",
            font=ctk.CTkFont(
                size=19,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=25,
            pady=(25, 8)
        )


        ctk.CTkLabel(
            card,
            text=(
                "Export all saved assessments into a readable text report."
            ),
            font=ctk.CTkFont(
                size=12
            ),
            text_color=SECONDARY_TEXT
        ).pack(
            anchor="w",
            padx=25,
            pady=(0, 20)
        )


        ctk.CTkButton(
            card,
            text="Export Assessment Report",
            command=self.export_report,
            width=220,
            height=42,
            corner_radius=9,
            fg_color=BLUE_COLOR,
            hover_color="#1D4ED8"
        ).pack(
            anchor="w",
            padx=25,
            pady=(0, 25)
        )


    def export_report(self):

        history = self.load_history()


        if not history:

            messagebox.showinfo(
                "No Data",
                "There are no assessments to export."
            )

            return


        file_path = filedialog.asksaveasfilename(
            title="Save Medical Assessment Report",
            defaultextension=".txt",
            filetypes=[
                (
                    "Text Files",
                    "*.txt"
                )
            ]
        )


        if not file_path:

            return


        try:

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "EARLY MEDICAL DIAGNOSTICS ENGINE\n"
                )

                file.write(
                    "CHRONIC DISEASE RISK ASSESSMENT REPORT\n"
                )

                file.write(
                    "=" * 60 + "\n\n"
                )


                for index, record in enumerate(
                    history,
                    start=1
                ):

                    patient = record.get(
                        "patient",
                        {}
                    )

                    predictions = record.get(
                        "predictions",
                        {}
                    )


                    file.write(
                        f"ASSESSMENT {index}\n"
                    )

                    file.write(
                        "-" * 50 + "\n"
                    )


                    file.write(
                        f"Date: {record.get('timestamp', '')}\n\n"
                    )


                    file.write(
                        "PATIENT INFORMATION\n"
                    )


                    for key, value in patient.items():

                        file.write(
                            f"{key}: {value}\n"
                        )


                    file.write(
                        "\nPREDICTIONS\n"
                    )


                    for disease in [
                        "diabetes",
                        "heart",
                        "kidney"
                    ]:

                        result = predictions.get(
                            disease,
                            {}
                        )


                        file.write(
                            f"{disease.title()}: "
                            f"{result.get('risk', 'N/A')} "
                            f"({result.get('probability', 0):.2f}%)\n"
                        )


                    file.write(
                        f"\nOverall Risk: "
                        f"{predictions.get('overall', 'N/A')}\n"
                    )


                    file.write(
                        "\n" + "=" * 60 + "\n\n"
                    )


                file.write(
                    "DISCLAIMER:\n"
                )

                file.write(
                    "This report is generated for academic/research purposes "
                    "and is not a substitute for professional medical diagnosis.\n"
                )


            messagebox.showinfo(
                "Report Exported",
                "The report was successfully exported."
            )


        except Exception as e:

            messagebox.showerror(
                "Export Error",
                str(e)
            )


    # ========================================================
    # SETTINGS PAGE
    # ========================================================

    def show_settings(self):

        page = self.create_page()


        ctk.CTkLabel(
            page,
            text="Settings",
            font=ctk.CTkFont(
                size=28,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=35,
            pady=(30, 5)
        )


        ctk.CTkLabel(
            page,
            text="Application and model information",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=SECONDARY_TEXT
        ).pack(
            anchor="w",
            padx=35
        )


        # ----------------------------------------------------
        # Appearance
        # ----------------------------------------------------

        appearance_card = ctk.CTkFrame(
            page,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        appearance_card.pack(
            fill="x",
            padx=35,
            pady=20
        )


        ctk.CTkLabel(
            appearance_card,
            text="Appearance",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=25,
            pady=(20, 10)
        )


        ctk.CTkLabel(
            appearance_card,
            text="Color Mode",
            text_color=SECONDARY_TEXT
        ).pack(
            anchor="w",
            padx=25
        )


        appearance = ctk.CTkOptionMenu(
            appearance_card,
            values=[
                "Light",
                "Dark",
                "System"
            ],
            command=self.change_appearance,
            width=160
        )

        appearance.set(
            "Light"
        )

        appearance.pack(
            anchor="w",
            padx=25,
            pady=(5, 20)
        )


        # ----------------------------------------------------
        # Model status
        # ----------------------------------------------------

        model_card = ctk.CTkFrame(
            page,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        model_card.pack(
            fill="x",
            padx=35,
            pady=10
        )


        ctk.CTkLabel(
            model_card,
            text="Machine Learning Models",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=25,
            pady=(20, 15)
        )


        self.create_model_status(
            model_card,
            "Diabetes Model",
            diabetes_model
        )

        self.create_model_status(
            model_card,
            "Heart Disease Model",
            heart_model
        )

        self.create_model_status(
            model_card,
            "Kidney Disease Model",
            kidney_model
        )


        # ----------------------------------------------------
        # Project information
        # ----------------------------------------------------

        project_card = ctk.CTkFrame(
            page,
            fg_color=CARD_COLOR,
            corner_radius=15
        )

        project_card.pack(
            fill="x",
            padx=35,
            pady=10
        )


        ctk.CTkLabel(
            project_card,
            text="Project Information",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            anchor="w",
            padx=25,
            pady=(20, 10)
        )


        info_text = (
            "Project Title:\n"
            "Early Medical Diagnostics Engine for Chronic Disease Risk Assessment\n\n"
            "Technology:\n"
            "Python, CustomTkinter, Pandas, Scikit-learn, Joblib\n\n"
            "Purpose:\n"
            "Machine-learning based early risk assessment for chronic diseases."
        )


        ctk.CTkLabel(
            project_card,
            text=info_text,
            font=ctk.CTkFont(
                size=12
            ),
            text_color=SECONDARY_TEXT,
            justify="left"
        ).pack(
            anchor="w",
            padx=25,
            pady=(0, 25)
        )


    def change_appearance(
        self,
        choice
    ):

        if choice == "Light":

            ctk.set_appearance_mode(
                "light"
            )

        elif choice == "Dark":

            ctk.set_appearance_mode(
                "dark"
            )

        else:

            ctk.set_appearance_mode(
                "system"
            )


    def create_model_status(
        self,
        parent,
        name,
        model
    ):

        status = (
            "Loaded Successfully"
            if model is not None
            else "Not Loaded"
        )


        status_color = (
            GREEN_COLOR
            if model is not None
            else RED_COLOR
        )


        row = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        row.pack(
            fill="x",
            padx=25,
            pady=5
        )


        ctk.CTkLabel(
            row,
            text=name,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color=TEXT_COLOR
        ).pack(
            side="left"
        )


        ctk.CTkLabel(
            row,
            text=status,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            text_color=status_color
        ).pack(
            side="right"
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app = MedicalDiagnosticsApp()

    app.mainloop()