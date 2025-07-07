from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models
model = joblib.load("MODELS/SP_model.pkl")
scaler = joblib.load("MODELS/SP_scaler.pkl")
columns = joblib.load("MODELS/SP_columns.pkl")
edu_encoder = joblib.load("MODELS/SP_edu_encoder.pkl")
job_titles = joblib.load("MODELS/SP_job_titles.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect form data
        age = int(request.form["age"])
        gender = 1 if request.form["gender"] == "Male" else 0
        edu = request.form["education"]
        job = request.form["job"]
        exp = float(request.form["experience"])

        # Encode education
        edu_encoded = edu_encoder.transform([[edu]])[0][0]

        # Create dataframe with default 0
        data_dict = {
            "Age": age,
            "Gender": gender,
            "Education Level": edu_encoded,
            "Years of Experience": exp
        }

        for col in columns:
            if col.startswith("Job Title_"):
                data_dict[col] = 1 if col == f"Job Title_{job}" else 0

        # Ensure all expected columns are present
        for col in columns:
            if col not in data_dict:
                data_dict[col] = 0

        df = pd.DataFrame([data_dict])[columns]

        # Scale and predict
        scaled_input = scaler.transform(df)
        salary = model.predict(scaled_input)[0]
        prediction = f"${salary:,.2f}"

    return render_template("index.html", prediction=prediction, job_titles=job_titles)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
