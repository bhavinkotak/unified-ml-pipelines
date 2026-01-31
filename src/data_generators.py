"""
ML Experiment Data Generators
==============================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

Generates synthetic datasets for ML pipeline testing and demonstration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass  
class DataGeneratorConfig:
    """Configuration for data generation"""
    random_seed: int = 42
    n_samples: int = 1000
    n_features: int = 10
    noise_level: float = 0.1
    class_imbalance: float = 0.5  # Fraction of positive class


class MLExperimentDataGenerator:
    """
    Generate synthetic datasets for ML experiments.
    
    Creates realistic datasets with:
    - Configurable feature correlations
    - Class imbalance options
    - Noise injection
    - Feature interactions
    """
    
    def __init__(self, random_seed: int = 42):
        self._rng = np.random.RandomState(random_seed)
        self.random_seed = random_seed
    
    def generate_classification_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_informative: int = 5,
        n_redundant: int = 2,
        class_weight: float = 0.5,
        noise: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate a classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features
            n_redundant: Number of redundant (correlated) features
            class_weight: Fraction of positive class (0-1)
            noise: Amount of noise to add
            
        Returns:
            DataFrame with features and target
        """
        # Generate informative features
        n_positive = int(n_samples * class_weight)
        n_negative = n_samples - n_positive
        
        # Positive class features
        X_pos = self._rng.randn(n_positive, n_informative) + 1.0
        
        # Negative class features
        X_neg = self._rng.randn(n_negative, n_informative) - 1.0
        
        # Combine
        X_informative = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
        
        # Shuffle
        shuffle_idx = self._rng.permutation(n_samples)
        X_informative = X_informative[shuffle_idx]
        y = y[shuffle_idx]
        
        # Generate redundant features (linear combinations)
        X_redundant = np.zeros((n_samples, n_redundant))
        for i in range(n_redundant):
            weights = self._rng.randn(n_informative)
            X_redundant[:, i] = X_informative @ weights
        
        # Generate noise features
        n_noise = n_features - n_informative - n_redundant
        X_noise = self._rng.randn(n_samples, max(0, n_noise))
        
        # Combine all features
        feature_arrays = [X_informative, X_redundant]
        if n_noise > 0:
            feature_arrays.append(X_noise)
        
        X = np.hstack(feature_arrays)
        
        # Add noise to all features
        X += noise * self._rng.randn(*X.shape)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y.astype(int)
        
        return df
    
    def generate_regression_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_informative: int = 5,
        noise: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate a regression dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features
            noise: Standard deviation of noise
            
        Returns:
            DataFrame with features and target
        """
        # Generate features
        X = self._rng.randn(n_samples, n_features)
        
        # Generate target as linear combination of informative features
        weights = self._rng.randn(n_informative) * 5
        y = X[:, :n_informative] @ weights
        
        # Add non-linear interactions
        y += 0.5 * X[:, 0] * X[:, 1]  # Interaction term
        y += 0.3 * np.square(X[:, 2])  # Quadratic term
        
        # Add noise
        y += noise * self._rng.randn(n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def generate_time_series_features(
        self,
        n_samples: int = 1000,
        n_periods: int = 30
    ) -> pd.DataFrame:
        """Generate time series feature data"""
        
        data = []
        base_date = datetime.now() - timedelta(days=n_samples)
        
        for i in range(n_samples):
            timestamp = base_date + timedelta(days=i)
            
            # Generate time-based features
            row = {
                'timestamp': timestamp,
                'day_of_week': timestamp.weekday(),
                'month': timestamp.month,
                'is_weekend': int(timestamp.weekday() >= 5),
                'rolling_mean': self._rng.uniform(50, 150),
                'rolling_std': self._rng.uniform(5, 25),
                'lag_1': self._rng.uniform(40, 160),
                'lag_7': self._rng.uniform(40, 160),
                'trend': i * 0.01 + self._rng.normal(0, 0.5),
                'seasonality': np.sin(2 * np.pi * i / 365) * 10
            }
            
            # Target based on features
            row['target'] = (
                row['rolling_mean'] * 0.5 +
                row['trend'] * 10 +
                row['seasonality'] +
                self._rng.normal(0, 5)
            )
            
            data.append(row)
        
        return pd.DataFrame(data)


class FeatureDataGenerator:
    """
    Generate synthetic feature store data.
    
    Creates datasets representing features from different
    data sources with realistic patterns.
    """
    
    def __init__(self, random_seed: int = 42):
        self._rng = np.random.RandomState(random_seed)
    
    def generate_customer_features(self, n_customers: int = 1000) -> pd.DataFrame:
        """Generate customer feature data"""
        
        data = {
            'entity_id': range(n_customers),
            'customer_age': self._rng.normal(45, 15, n_customers).clip(18, 85).astype(int),
            'account_tenure_months': self._rng.exponential(36, n_customers).clip(1, 240).astype(int),
            'total_transactions': self._rng.poisson(100, n_customers),
            'avg_transaction_amount': self._rng.lognormal(4, 1, n_customers).round(2),
            'credit_score': self._rng.normal(700, 80, n_customers).clip(300, 850).astype(int),
            'num_products': self._rng.poisson(3, n_customers).clip(1, 10),
            'is_premium': (self._rng.random(n_customers) > 0.8).astype(int),
            'email_engagement': self._rng.uniform(0, 1, n_customers).round(3),
            'churn_risk_score': self._rng.beta(2, 5, n_customers).round(3)
        }
        
        return pd.DataFrame(data)
    
    def generate_transaction_features(self, n_transactions: int = 5000) -> pd.DataFrame:
        """Generate transaction feature data"""
        
        data = {
            'entity_id': self._rng.randint(0, 1000, n_transactions),
            'transaction_amount': self._rng.lognormal(4, 1.5, n_transactions).round(2),
            'hour_of_day': self._rng.randint(0, 24, n_transactions),
            'day_of_week': self._rng.randint(0, 7, n_transactions),
            'is_online': (self._rng.random(n_transactions) > 0.3).astype(int),
            'merchant_category': self._rng.choice(
                ['grocery', 'restaurant', 'retail', 'travel', 'entertainment'],
                n_transactions
            ),
            'distance_from_home': self._rng.exponential(10, n_transactions).round(2),
            'velocity_1h': self._rng.poisson(2, n_transactions),
            'velocity_24h': self._rng.poisson(8, n_transactions),
            'amount_zscore': self._rng.normal(0, 1, n_transactions).round(3)
        }
        
        return pd.DataFrame(data)
    
    def generate_product_features(self, n_products: int = 500) -> pd.DataFrame:
        """Generate product feature data"""
        
        categories = ['electronics', 'clothing', 'home', 'sports', 'books', 'food']
        
        data = {
            'entity_id': range(n_products),
            'category': self._rng.choice(categories, n_products),
            'price': self._rng.lognormal(3, 1, n_products).round(2),
            'inventory_level': self._rng.poisson(50, n_products),
            'avg_rating': self._rng.uniform(1, 5, n_products).round(2),
            'num_reviews': self._rng.poisson(100, n_products),
            'days_since_last_sale': self._rng.exponential(7, n_products).astype(int),
            'conversion_rate': self._rng.beta(2, 10, n_products).round(4),
            'return_rate': self._rng.beta(1, 20, n_products).round(4),
            'profit_margin': self._rng.uniform(0.1, 0.6, n_products).round(3)
        }
        
        return pd.DataFrame(data)


# Main execution for testing
if __name__ == "__main__":
    print("Testing Data Generators...")
    
    # Test ML experiment data
    ml_gen = MLExperimentDataGenerator(random_seed=42)
    
    clf_data = ml_gen.generate_classification_dataset(n_samples=500, n_features=10)
    print(f"Classification dataset: {clf_data.shape}")
    print(f"Target distribution: {clf_data['target'].value_counts().to_dict()}")
    
    reg_data = ml_gen.generate_regression_dataset(n_samples=500, n_features=8)
    print(f"Regression dataset: {reg_data.shape}")
    print(f"Target range: [{reg_data['target'].min():.2f}, {reg_data['target'].max():.2f}]")
    
    # Test feature data
    feat_gen = FeatureDataGenerator(random_seed=42)
    
    customer_data = feat_gen.generate_customer_features(n_customers=200)
    print(f"Customer features: {customer_data.shape}")
    
    transaction_data = feat_gen.generate_transaction_features(n_transactions=1000)
    print(f"Transaction features: {transaction_data.shape}")
    
    print("\nAll data generators working correctly!")
