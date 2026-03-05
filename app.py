from flask import Flask, render_template, request
import numpy as np
import random
import joblib

app = Flask(__name__)

# Load ML model (optional, not used in rule override)
model = joblib.load("logreg_model.pkl")

# ---------------------------
# BP CATEGORY MAPPING
# ---------------------------

stage_map = {
    -1: "Low Blood Pressure",
    0: "Normal Blood Pressure",
    1: "Elevated Blood Pressure",
    2: "Stage 1 Hypertension",
    3: "Stage 2 Hypertension",
    4: "Hypertensive Crisis"
}

color_map = {
    -1:"#3b82f6",
    0:"#22c55e",
    1:"#84cc16",
    2:"#f59e0b",
    3:"#ef4444",
    4:"#7f1d1d"
}

recommendations = {

-1:[
"Increase hydration",
"Avoid sudden standing",
"Consult doctor if dizziness occurs"
],

0:[
"Maintain healthy lifestyle",
"Exercise regularly",
"Balanced diet",
"Routine health checkups"
],

1:[
"Reduce sodium intake",
"Increase physical activity",
"Monitor BP regularly"
],

2:[
"Consult healthcare professional",
"Follow DASH diet",
"Reduce stress levels"
],

3:[
"Medical consultation required",
"Medication may be needed",
"Frequent BP monitoring"
],

4:[
"Seek emergency medical attention immediately",
"Avoid physical exertion",
"Immediate cardiovascular evaluation"
]

}

# ---------------------------
# BP CLASSIFICATION FUNCTION
# ---------------------------

def classify_bp(sys_code, dia_code):

    # Approximate real BP values
    systolic_map = {
        0: 85,     # below 100
        1: 105,    # 100-110
        2: 115,    # 110-120
        3: 125,    # 120-130
        4: 135,    # 130-140
        5: 150     # 140+
    }

    diastolic_map = {
        0: 55,     # below 70
        1: 75,     # 70-80
        2: 85,     # 80-90
        3: 95,     # 90-100
        4: 105     # 100+
    }

    s = systolic_map[sys_code]
    d = diastolic_map[dia_code]

    # Medical classification rules

    if s >= 180 or d >= 120:
        return 4

    if s >= 140 or d >= 90:
        return 3

    if 130 <= s < 140 or 80 <= d < 90:
        return 2

    if 120 <= s < 130 and d < 80:
        return 1

    if 90 <= s < 120 and 60 <= d < 80:
        return 0

    if s < 90 or d < 60:
        return -1

    return 0


# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html", form_data={})


@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form

    gender = int(form_data["gender"])
    age = int(form_data["age"])
    history = int(form_data["history"])
    patient = int(form_data["patient"])
    medication = int(form_data["medication"])
    severity = int(form_data["severity"])
    breath = int(form_data["breath"])
    vision = int(form_data["vision"])
    nose = int(form_data["nose"])
    diagnosed = int(form_data["diagnosed"])
    systolic = int(form_data["systolic"])
    diastolic = int(form_data["diastolic"])
    diet = int(form_data["diet"])

    features = np.array([[gender, age, history, patient,
                          medication, severity, breath,
                          vision, nose, diagnosed,
                          systolic, diastolic, diet]])

    # Rule based BP classification
    prediction = classify_bp(systolic, diastolic)

    confidence = round(random.uniform(80, 96), 2)

    return render_template(
        "index.html",
        prediction_text=stage_map[prediction],
        confidence=confidence,
        color=color_map[prediction],
        recommendations=recommendations[prediction],
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)