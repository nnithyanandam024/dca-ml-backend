"""
Feature engineering for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def encode_categorical(self, df, fit=True):
        """
        Encode categorical features
        Safely handles unseen labels during inference
        """
        categorical_features = [
            'segment',
            'industry',
            'payment_history',
            'geographic_region'
        ]

        df_encoded = df.copy()

        for feature in categorical_features:
            if fit:
                le = LabelEncoder()
                df_encoded[f'{feature}_encoded'] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
            else:
                le = self.label_encoders[feature]

                # ✅ SAFE handling for unseen labels
                known_classes = set(le.classes_)
                fallback_class = le.classes_[0]

                safe_values = df[feature].apply(
                    lambda x: x if x in known_classes else fallback_class
                )

                df_encoded[f'{feature}_encoded'] = le.transform(safe_values)

        return df_encoded

    def create_features(self, df):
        """Create engineered features"""
        df_features = df.copy()

        # Amount buckets
        df_features['amount_bucket'] = pd.cut(
            df['amount'],
            bins=[0, 5000, 25000, 100000, float('inf')],
            labels=[1, 2, 3, 4]
        ).astype(int)

        # Aging buckets
        df_features['aging_bucket'] = pd.cut(
            df['aging_days'],
            bins=[0, 30, 60, 90, 180, float('inf')],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Interaction features
        df_features['amount_aging_ratio'] = df['amount'] / (df['aging_days'] + 1)
        df_features['contact_efficiency'] = df['contact_attempts'] / (df['aging_days'] + 1)

        # Risk score
        df_features['combined_risk_score'] = (
            df_features['industry_risk_score'] * 0.4 +
            df_features['aging_bucket'] * 0.3 +
            (5 - df_features['customer_value_score'] / 20) * 0.3
        )

        return df_features

    def get_model_features(self, df, feature_set='recovery'):
        """Get features for specific model"""
        if feature_set == 'recovery':
            return [
                'amount',
                'aging_days',
                'contact_attempts',
                'days_since_last_contact',
                'customer_value_score',
                'industry_risk_score',
                'segment_encoded',
                'industry_encoded',
                'payment_history_encoded',
                'geographic_region_encoded',
                'amount_bucket',
                'aging_bucket',
                'amount_aging_ratio',
                'contact_efficiency',
                'combined_risk_score'
            ]

        elif feature_set == 'priority':
            return [
                'amount',
                'aging_days',
                'contact_attempts',
                'payment_history_encoded',
                'segment_encoded',
                'industry_encoded',
                'customer_value_score',
                'aging_bucket',
                'amount_bucket',
                'combined_risk_score'
            ]

        elif feature_set == 'amount':
            return [
                'amount',
                'aging_days',
                'payment_history_encoded',
                'segment_encoded',
                'industry_encoded',
                'customer_value_score',
                'contact_attempts',
                'industry_risk_score',
                'amount_aging_ratio'
            ]

        else:
            raise ValueError(f"Unknown feature set: {feature_set}")

    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
