"""
Generate 500 FedEx-realistic training cases
This matches the JavaScript data generation but for Python
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

FEDEX_STATS = {
    'customer_segments': {
        'enterprise': {'weight': 0.15, 'amount_range': (50000, 500000)},
        'mid_market': {'weight': 0.35, 'amount_range': (10000, 100000)},
        'small_business': {'weight': 0.50, 'amount_range': (2000, 30000)}
    },
    'industries': {
        'healthcare': 0.20,
        'technology': 0.18,
        'retail': 0.25,
        'manufacturing': 0.15,
        'automotive': 0.12,
        'other': 0.10
    },
    'payment_history_by_segment': {
        'enterprise': [0.5, 0.3, 0.15, 0.05],  # excellent, good, fair, poor
        'mid_market': [0.25, 0.40, 0.25, 0.10],
        'small_business': [0.15, 0.30, 0.35, 0.20]
    }
}

COMPANY_NAMES = {
    'healthcare': ['HealthCare Systems', 'Medical Supplies Co', 'MediTech Partners', 'Pharma Logistics', 'Clinical Services'],
    'technology': ['Tech Solutions Inc', 'Digital Innovations', 'Software Systems', 'Cloud Services Co', 'Data Analytics'],
    'retail': ['Retail Dynamics', 'E-Commerce Solutions', 'Fashion Outlets', 'Consumer Goods Inc', 'Online Marketplace'],
    'manufacturing': ['Manufacturing Systems', 'Industrial Components', 'Production Facilities', 'Assembly Solutions'],
    'automotive': ['Auto Parts Distributors', 'Vehicle Systems', 'Automotive Solutions', 'Parts Warehouse'],
    'other': ['General Business LLC', 'Commercial Services', 'Trading Company', 'Enterprise Solutions']
}

def generate_fedex_dataset(n_samples=500):
    """Generate realistic FedEx debt collection dataset"""
    
    cases = []
    today = datetime.now()
    
    for i in range(n_samples):
        # Select segment
        segment = np.random.choice(
            list(FEDEX_STATS['customer_segments'].keys()),
            p=[0.15, 0.35, 0.50]
        )
        
        # Select industry
        industry = np.random.choice(
            list(FEDEX_STATS['industries'].keys()),
            p=list(FEDEX_STATS['industries'].values())
        )
        
        # Generate amount
        amount_min, amount_max = FEDEX_STATS['customer_segments'][segment]['amount_range']
        amount = np.random.lognormal(
            mean=np.log(amount_min + amount_max) / 2,
            sigma=0.7
        )
        amount = np.clip(amount, amount_min, amount_max)
        
        # Aging (days overdue)
        aging = int(np.random.gamma(shape=3, scale=20))
        
        # Payment history
        payment_histories = ['excellent', 'good', 'fair', 'poor']
        payment_probs = FEDEX_STATS['payment_history_by_segment'][segment]
        payment_history = np.random.choice(payment_histories, p=payment_probs)
        
        # Calculate ground truth recovery probability
        base_prob = {'excellent': 0.92, 'good': 0.75, 'fair': 0.55, 'poor': 0.30}[payment_history]
        aging_penalty = min(aging / 365, 0.4)
        segment_bonus = {'enterprise': 0.05, 'mid_market': 0.02, 'small_business': 0}[segment]
        recovery_probability = max(base_prob - aging_penalty + segment_bonus, 0.1)
        recovery_probability = min(recovery_probability, 0.98)
        
        # Priority (ground truth)
        if aging > 90 and amount > 50000:
            priority = 'critical'
        elif aging > 60 or amount > 75000:
            priority = 'high'
        elif aging > 30 or amount > 25000:
            priority = 'medium'
        else:
            priority = 'low'
        
        # Status
        status = np.random.choice(
            ['new', 'assigned', 'in-progress', 'resolved', 'escalated'],
            p=[0.15, 0.20, 0.30, 0.20, 0.15]
        )
        
        # Contact attempts
        contact_attempts = 0 if status == 'new' else max(1, int(aging / 15))
        
        # Days since last contact
        days_since_last_contact = np.random.randint(1, min(aging, 60)) if status != 'new' else aging
        
        # Geographic region
        geographic = 'US_Domestic' if np.random.random() < 0.68 else 'International'
        
        # Customer lifetime value score (1-100)
        customer_value_score = int(np.random.beta(2, 5) * 100)
        
        # Industry risk score (1-5)
        industry_risk = {
            'healthcare': 2, 'technology': 3, 'retail': 4,
            'manufacturing': 3, 'automotive': 3, 'other': 3
        }[industry]
        
        # Expected recovery amount
        expected_recovery = amount * recovery_probability
        
        # Actual recovered (for resolved cases)
        if status == 'resolved':
            actual_recovered = expected_recovery * np.random.uniform(0.9, 1.1)
        else:
            actual_recovered = 0
        
        case = {
            'case_id': f'FDX-{i+1:05d}',
            'customer_name': np.random.choice(COMPANY_NAMES[industry]),
            'segment': segment,
            'industry': industry,
            'amount': round(amount, 2),
            'aging_days': aging,
            'payment_history': payment_history,
            'contact_attempts': contact_attempts,
            'days_since_last_contact': days_since_last_contact,
            'geographic_region': geographic,
            'customer_value_score': customer_value_score,
            'industry_risk_score': industry_risk,
            'status': status,
            'priority': priority,
            'recovery_probability': round(recovery_probability, 4),
            'expected_recovery': round(expected_recovery, 2),
            'actual_recovered': round(actual_recovered, 2) if status == 'resolved' else None
        }
        
        cases.append(case)
    
    df = pd.DataFrame(cases)
    return df

if __name__ == '__main__':
    # Generate dataset
    df = generate_fedex_dataset(500)
    
    # Save to CSV
    df.to_csv('fedex_cases.csv', index=False)
    
    print("✅ Generated 500 FedEx-realistic cases")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSegment distribution:\n{df['segment'].value_counts()}")
    print(f"\nIndustry distribution:\n{df['industry'].value_counts()}")
    print(f"\nPriority distribution:\n{df['priority'].value_counts()}")
    print(f"\nTotal outstanding: ${df['amount'].sum():,.2f}")
    print(f"Expected recovery: ${df['expected_recovery'].sum():,.2f}")
    print(f"Average recovery rate: {df['recovery_probability'].mean():.2%}")