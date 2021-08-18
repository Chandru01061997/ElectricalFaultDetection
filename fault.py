from flask import Flask, app, render_template, request
from logging import debug
from flask.templating import render_template
import pandas as pd
import joblib

app  = Flask(__name__)
model = joblib.load('electric_fault_detect_model.pkl')

@app.route("/")
def form():
    return render_template('fault.html')

@app.route("/electricfaultdetection", methods=['POST'])
def heartfailuredetection():

    Ia = request.form.get("Current-'a'phase")
    Ib = request.form.get("Current-'b'phase")
    Ic = request.form.get("Current-'c'phase")
    Va = request.form.get("Voltage-'a'phase")
    Vb = request.form.get("Voltage-'b'phase")
    Vc = request.form.get("Voltage-'c'phase")

    prediction = model.predict([[Ia, Ib, Ic, Va, Vb, Vc]])
    output = prediction[0]

    return render_template('fault.html' , final_output = f"The Fault occured is - {output} Fault")

if __name__ == "__main__":
    app.run(debug=True)
