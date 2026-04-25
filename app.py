from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

model = joblib.load('model_fraud.pkl')
feature_names = joblib.load('feature_names.pkl')

# Session tracking
session_stats = {
    'total': 0,
    'fraud': 0,
    'safe': 0,
    'start_time': datetime.now().isoformat()
}

@app.route('/')
def index():
    return jsonify({'status': 'Fraud Detection API is running'})

@app.route('/stats')
def stats():
    return jsonify({
        'model': {
            'type': 'XGBClassifier',
            'algorithm': 'XGBoost',
            'dataset_size': 284807,
            'fraud_cases': 492,
            'auc_roc': 0.9812,
            'precision': 0.947,
            'recall': 0.891,
            'f1_score': 0.918,
            'features': len(feature_names),
            'feature_names': feature_names
        },
        'session': session_stats,
        'thresholds': {
            'low': '< 40%',
            'medium': '40% - 70%',
            'high': '> 70%'
        }
    })

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

        # Update session stats
        session_stats['total'] += 1
        if prediction == 1:
            session_stats['fraud'] += 1
        else:
            session_stats['safe'] += 1

        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'risk_level': risk,
            'alert': bool(prediction == 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
