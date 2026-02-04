"""Tests for data preprocessing and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessor import DataPreprocessor, FeatureEngineer


class TestDataPreprocessor:
    """Test suite for DataPreprocessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = pd.DataFrame({
            'income': [50000, 75000, np.nan, 100000],
            'loan_amount': [150000, 250000, 200000, 300000],
            'credit_score': [650, 750, 700, 800],
            'employment_years': [3, 5, 2, 10],
            'age': [35, 45, 30, 55],
            'education_level': ['Bachelor', 'Master', 'Bachelor', 'PhD'],
            'marital_status': ['Married', 'Single', np.nan, 'Married']
        })
        return data
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert not preprocessor.is_fitted
        assert preprocessor.scaler is None
        assert len(preprocessor.label_encoders) == 0
    
    def test_fit_preprocessor(self, sample_data):
        """Test fitting preprocessor."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data)
        
        assert preprocessor.is_fitted
        assert preprocessor.scaler is not None
        assert 'education_level' in preprocessor.label_encoders
        assert 'marital_status' in preprocessor.label_encoders
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError):
            preprocessor.transform(sample_data)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_data)
        
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]
        assert not result.isnull().any().any()
    
    def test_handle_missing_values_mean(self, sample_data):
        """Test missing value handling with mean strategy."""
        result = DataPreprocessor.handle_missing_values(sample_data, strategy='mean')
        
        assert not result.isnull().any().any()
        # Check income was imputed with mean
        assert result.loc[2, 'income'] > 0
    
    def test_handle_missing_values_median(self, sample_data):
        """Test missing value handling with median strategy."""
        result = DataPreprocessor.handle_missing_values(sample_data, strategy='median')
        
        assert not result.isnull().any().any()
    
    def test_handle_missing_values_invalid_strategy(self, sample_data):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError):
            DataPreprocessor.handle_missing_values(sample_data, strategy='invalid')
    
    def test_scaling_numerical_features(self, sample_data):
        """Test that numerical features are scaled."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_data)
        
        # Check scaled values (should be roughly -3 to 3 range)
        numerical_cols = ['income', 'loan_amount', 'credit_score', 'employment_years', 'age']
        for col in numerical_cols:
            if result[col].notna().any():
                assert result[col].max() <= 5  # Scaled values should be small
                assert result[col].min() >= -5


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = pd.DataFrame({
            'income': [50000, 75000, 100000, 120000],
            'loan_amount': [150000, 250000, 200000, 300000],
            'credit_score': [650, 750, 700, 800],
            'employment_years': [3, 5, 2, 10],
            'age': [35, 45, 30, 55],
            'education_level': ['Bachelor', 'Master', 'Bachelor', 'PhD'],
            'marital_status': ['Married', 'Single', 'Single', 'Married']
        })
        return data
    
    def test_income_to_loan_ratio(self, sample_data):
        """Test income-to-loan ratio calculation."""
        ratio = FeatureEngineer.create_income_to_loan_ratio(sample_data)
        
        assert len(ratio) == len(sample_data)
        assert (ratio > 0).all()
        # Higher income should give higher ratio
        assert ratio.iloc[3] > ratio.iloc[0]  # income 120k vs 50k
    
    def test_credit_history_score(self, sample_data):
        """Test credit history score calculation."""
        score = FeatureEngineer.create_credit_history_score(sample_data)
        
        assert len(score) == len(sample_data)
        assert (score >= 0).all()
        assert (score <= 1).all()  # Should be normalized
    
    def test_employment_stability(self, sample_data):
        """Test employment stability calculation."""
        stability = FeatureEngineer.create_employment_stability(sample_data)
        
        assert len(stability) == len(sample_data)
        assert (stability >= 0).all()
        assert (stability <= 1).all()
        # More years should give higher stability
        assert stability.iloc[3] > stability.iloc[0]  # 10 years vs 3 years
    
    def test_age_credit_interaction(self, sample_data):
        """Test age-credit score interaction."""
        interaction = FeatureEngineer.create_age_credit_interaction(sample_data)
        
        assert len(interaction) == len(sample_data)
        assert (interaction >= 0).all()
    
    def test_loan_amount_category(self, sample_data):
        """Test loan amount categorization."""
        category = FeatureEngineer.create_loan_amount_category(sample_data)
        
        assert len(category) == len(sample_data)
        assert category.isin([0, 1, 2, 3]).all()
    
    def test_income_category(self, sample_data):
        """Test income categorization."""
        category = FeatureEngineer.create_income_category(sample_data)
        
        assert len(category) == len(sample_data)
        assert category.isin([0, 1, 2, 3]).all()
    
    def test_engineer_features_creates_all_features(self, sample_data):
        """Test that engineer_features creates all engineered features."""
        engineered = FeatureEngineer.engineer_features(sample_data)
        
        expected_features = [
            'income_to_loan_ratio',
            'credit_history_score',
            'employment_stability',
            'age_credit_interaction',
            'loan_amount_category',
            'income_category'
        ]
        
        for feature in expected_features:
            assert feature in engineered.columns
            assert not engineered[feature].isnull().any()


class TestIntegration:
    """Integration tests for preprocessing pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'income': np.random.randint(40000, 150000, n_samples),
            'loan_amount': np.random.randint(100000, 400000, n_samples),
            'credit_score': np.random.randint(600, 850, n_samples),
            'employment_years': np.random.randint(0, 30, n_samples),
            'age': np.random.randint(25, 65, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
        })
        return data
    
    def test_full_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        # Engineer features
        engineered = FeatureEngineer.engineer_features(sample_data)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        processed = preprocessor.fit_transform(engineered)
        
        # Verify output
        assert processed.shape[0] == engineered.shape[0]
        assert not processed.isnull().any().any()
        assert processed.dtypes.apply(lambda x: x in [np.float64, np.int64]).all()
    
    def test_train_test_consistency(self, sample_data):
        """Test preprocessing consistency between train and test data."""
        # Split data
        train_idx = sample_data.sample(frac=0.7, random_state=42).index
        test_idx = sample_data.drop(train_idx).index
        
        X_train = sample_data.loc[train_idx]
        X_test = sample_data.loc[test_idx]
        
        # Fit on train data
        preprocessor = DataPreprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Apply to test data
        X_test_processed = preprocessor.transform(X_test)
        
        # Both should have same number of columns
        assert X_train_processed.shape[1] == X_test_processed.shape[1]
        
        # Test data should be properly transformed
        assert not X_test_processed.isnull().any().any()
