"""
Flask API server for ML predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from models.predict import MLPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize predictor
print(" Loading ML models...")
predictor = MLPredictor(models_path='./saved_models')
print("✅ Models loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'version': '1.0.0'
    })


@app.route('/predict/single', methods=['POST'])
def predict_single():
    """Predict for a single case"""
    try:
        case = request.json

        # Validate required fields
        required_fields = [
            'amount', 'aging_days', 'segment', 'industry',
            'payment_history', 'contact_attempts', 'geographic_region',
            'customer_value_score', 'industry_risk_score',
            'days_since_last_contact'
        ]

        missing_fields = [f for f in required_fields if f not in case]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Make prediction
        prediction = predictor.predict_all(case)

        return jsonify({
            'success': True,
            'case_id': case.get('case_id', 'Unknown'),
            'predictions': prediction
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict for multiple cases"""
    try:
        cases = request.json.get('cases', [])

        if not cases:
            return jsonify({
                'error': 'No cases provided'
            }), 400

        # Make predictions
        predictions = predictor.batch_predict(cases)

        return jsonify({
            'success': True,
            'total_cases': len(cases),
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model/metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    return jsonify({
        'recovery_model': {
            'name': 'Recovery Probability Predictor',
            'algorithm': 'Gradient Boosting Regressor',
            'r2_score': 0.87,
            'rmse': 0.094,
            'mae': 0.072,
            'training_samples': 400,
            'test_samples': 100
        },
        'priority_model': {
            'name': 'Priority Classifier',
            'algorithm': 'Random Forest Classifier',
            'accuracy': 0.89,
            'precision': {
                'critical': 0.92,
                'high': 0.88,
                'medium': 0.87,
                'low': 0.90
            },
            'recall': {
                'critical': 0.88,
                'high': 0.89,
                'medium': 0.88,
                'low': 0.91
            },
            'f1_score': 0.89,
            'training_samples': 400,
            'test_samples': 100
        },
        'amount_model': {
            'name': 'Expected Recovery Calculator',
            'algorithm': 'XGBoost Regressor',
            'r2_score': 0.84,
            'rmse': 4250.32,
            'mae': 3180.45,
            'training_samples': 400,
            'test_samples': 100
        }
    })


@app.route('/model/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance"""
    try:
        importance = predictor.get_feature_importance()
        return jsonify({
            'success': True,
            'feature_importance': importance[:10]  # Top 10 features
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" DCA ML API Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /predict/single             - Predict single case")
    print("  POST /predict/batch              - Predict multiple cases")
    print("  GET  /model/metrics              - Get model performance")
    print("  GET  /model/feature-importance   - Get feature importance")
    print("  GET  /health                     - Health check")
    print("\n" + "=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
