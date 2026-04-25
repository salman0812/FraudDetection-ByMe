from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load('model_fraud.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def index():
    return jsonify({'status': 'Fraud Detection API is running'})

@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data], columns=feature_names)
        input_df = input_df.fillna(0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if probability >= 0.7:
            risk = 'HIGH'
        elif probability >= 0.4:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'risk_level': risk,
            'alert': bool(prediction == 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
