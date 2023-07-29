from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('svc.pkl')

    data = {
        'Pregnancies': float(request.form.get('Pregnancies')),
        'Glucose': float(request.form.get('Glucose')),
        'BloodPressure': float(request.form.get('BloodPressure')),
        'SkinThickness': float(request.form.get('SkinThickness')),
        'Insulin': float(request.form.get('Insulin')),
        'BMI': float(request.form.get('BMI')),
        'DiabetesPedigreeFunction': float(request.form.get('DiabetesPedigreeFunction')),
        'Age': float(request.form.get('Age'))
    }

    features = np.array([list(data.values())])
    prediction_proba = model.predict_proba(features.reshape(1, -1))

    return f"The probability of the patient having diabetes is {prediction_proba[0][1]*100:.2f}%."

if __name__ == '__main__':
    app.run(debug=True)