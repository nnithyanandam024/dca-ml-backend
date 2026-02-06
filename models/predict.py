"""
models/predict.py
Prediction functions using trained models
"""

import pandas as pd
import numpy as np
import joblib


class MLPredictor:
    def __init__(self, models_path='../saved_models'):
        """Load all trained models"""
        self.recovery_model = joblib.load(f'{models_path}/recovery_model.pkl')
        self.priority_model = joblib.load(f'{models_path}/priority_model.pkl')
        self.amount_model = joblib.load(f'{models_path}/amount_model.pkl')
        self.feature_engineer = joblib.load(f'{models_path}/feature_engineer.pkl')
        self.recovery_importance = joblib.load(f'{models_path}/recovery_importance.pkl')

    def prepare_case(self, case_dict):
        """Convert case dictionary to DataFrame with required features"""
        df = pd.DataFrame([case_dict])

        df_encoded = self.feature_engineer.encode_categorical(df, fit=False)
        df_features = self.feature_engineer.create_features(df_encoded)

        return df_features

    def predict_recovery_probability(self, case):
        df = self.prepare_case(case)
        features = self.feature_engineer.get_model_features(df, 'recovery')
        X = df[features]
        X_scaled = self.feature_engineer.scale_features(X, fit=False)

        probability = self.recovery_model.predict(X_scaled)[0]
        return float(np.clip(probability, 0, 1))

    def predict_priority(self, case):
        df = self.prepare_case(case)
        features = self.feature_engineer.get_model_features(df, 'priority')
        X = df[features]

        priority = self.priority_model.predict(X)[0]
        probabilities = self.priority_model.predict_proba(X)[0]

        return {
            'priority': priority,
            'confidence': float(max(probabilities)),
            'probabilities': dict(zip(self.priority_model.classes_,
                                      map(float, probabilities)))
        }

    def predict_recovery_amount(self, case):
        df = self.prepare_case(case)
        features = self.feature_engineer.get_model_features(df, 'amount')
        X = df[features]

        amount = self.amount_model.predict(X)[0]
        return float(max(0, amount))

    def predict_all(self, case):
        recovery_prob = self.predict_recovery_probability(case)
        priority_result = self.predict_priority(case)
        recovery_amount = self.predict_recovery_amount(case)

        amount_score = min(case['amount'] / 100000, 1)      # 0–1
        aging_score = min(case['aging_days'] / 180, 1)      # 0–1

        ai_priority_score = (
          recovery_prob * 40 +
          amount_score * 20 +
          aging_score * 20 +
          priority_result['confidence'] * 20
       ) 
        ai_priority_score = min(100, max(0, ai_priority_score))

        return {
            'recovery_probability': round(recovery_prob, 4),
            'priority': priority_result['priority'],
            'priority_confidence': round(priority_result['confidence'], 4),
            'expected_recovery': round(recovery_amount, 2),
            'ai_priority_score': round(ai_priority_score, 2),
            'recommendation': self.generate_recommendation(
                case, recovery_prob, priority_result['priority']
            ),
            'reasoning': self.generate_reasoning(case, recovery_prob)
        }

    def generate_recommendation(self, case, recovery_prob, priority):
        aging = case['aging_days']
        amount = case['amount']

        if recovery_prob > 0.8 and aging < 30:
            return "Standard follow-up via email within 24 hours"
        elif recovery_prob > 0.6 and aging < 60:
            return "Phone contact within 48 hours with payment plan options"
        elif recovery_prob > 0.4 and priority in ['high', 'critical']:
            return "Immediate DCA assignment with escalation protocol"
        elif aging > 120:
            return "Consider legal action or write-off evaluation"
        elif amount > 100000:
            return "Executive-level intervention and customized recovery strategy"
        else:
            return "Intensive collection effort with weekly follow-ups required"

    def generate_reasoning(self, case, recovery_prob):
        factors = []

        if case['payment_history'] in ['excellent', 'good']:
            factors.append("strong payment history")
        if case['amount'] > 50000:
            factors.append("high-value account")
        if case['aging_days'] < 60:
            factors.append("recent overdue")
        if case['segment'] == 'enterprise':
            factors.append("enterprise customer")
        if case['customer_value_score'] > 70:
            factors.append("high customer lifetime value")
        if recovery_prob > 0.7:
            factors.append("strong recovery indicators")

        if not factors:
            return "Standard recovery profile based on historical patterns"

        return (
            f"ML model predicts {recovery_prob*100:.1f}% recovery due to: "
            f"{', '.join(factors)}"
        )

    def batch_predict(self, cases_list):
        results = []

        for case in cases_list:
            try:
                prediction = self.predict_all(case)
                results.append({
                    'case_id': case.get('case_id', 'Unknown'),
                    **prediction
                })
            except Exception as e:
                results.append({
                    'case_id': case.get('case_id', 'Unknown'),
                    'error': str(e)
                })

        return results

    def get_feature_importance(self):
        return self.recovery_importance.to_dict('records')
