"""Data preprocessing and feature engineering utilities."""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Handles data preprocessing and feature engineering for fraud detection.
    
    This class provides comprehensive preprocessing including handling missing values,
    encoding categorical variables, scaling numerical features, and creating
    engineered features like income-to-loan ratios and credit history scores.
    """
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer: Optional[SimpleImputer] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Training features
            
        Returns:
            Self for method chaining
        """
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit scaler for numerical features
        self.scaler = StandardScaler()
        self.scaler.fit(X[numerical_cols])
        
        # Fit label encoders for categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Fit imputer for missing values
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer.fit(X[numerical_cols])
        
        self.feature_names = numerical_cols + categorical_cols
        self.is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If preprocessor not fitted or columns mismatch
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        X_processed = X.copy()
        
        # Identify numerical and categorical columns
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values in numerical columns
        if numerical_cols and self.imputer:
            X_processed[numerical_cols] = self.imputer.transform(X_processed[numerical_cols])
        
        # Scale numerical features
        if numerical_cols and self.scaler:
            X_processed[numerical_cols] = self.scaler.transform(X_processed[numerical_cols])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in self.label_encoders:
                X_processed[col] = self.label_encoders[col].transform(
                    X_processed[col].astype(str)
                )
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)
    
    @staticmethod
    def handle_missing_values(X: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            X: Input data
            strategy: Imputation strategy ('mean', 'median', 'forward_fill')
            
        Returns:
            DataFrame with missing values handled
        """
        X_filled = X.copy()
        
        if strategy == 'mean':
            numerical_cols = X_filled.select_dtypes(include=[np.number]).columns
            X_filled[numerical_cols] = X_filled[numerical_cols].fillna(
                X_filled[numerical_cols].mean()
            )
        elif strategy == 'median':
            numerical_cols = X_filled.select_dtypes(include=[np.number]).columns
            X_filled[numerical_cols] = X_filled[numerical_cols].fillna(
                X_filled[numerical_cols].median()
            )
        elif strategy == 'forward_fill':
            X_filled = X_filled.fillna(method='ffill').fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Fill categorical columns with mode
        categorical_cols = X_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0])
        
        return X_filled


class FeatureEngineer:
    """
    Feature engineering for fraud detection model.
    
    Creates domain-specific features including income-to-loan ratios,
    credit history scores, and employment stability indicators.
    """
    
    @staticmethod
    def create_income_to_loan_ratio(X: pd.DataFrame) -> pd.Series:
        """
        Create income-to-loan ratio feature.
        
        Represents the proportion of income required to cover loan amount.
        Lower values indicate higher debt burden and increased fraud risk.
        
        Args:
            X: DataFrame with 'income' and 'loan_amount' columns
            
        Returns:
            Series with ratio values
        """
        return X['income'] / (X['loan_amount'] + 1)  # +1 to avoid division by zero
    
    @staticmethod
    def create_credit_history_score(X: pd.DataFrame) -> pd.Series:
        """
        Create credit history score based on multiple factors.
        
        Combines credit score with payment indicators to create a
        comprehensive credit risk score.
        
        Args:
            X: DataFrame with 'credit_score' column
            
        Returns:
            Series with credit history scores (0-1 normalized)
        """
        # Normalize credit score to 0-1 range (typical range: 300-850)
        credit_normalized = (X['credit_score'] - 300) / (850 - 300)
        credit_normalized = credit_normalized.clip(0, 1)
        return credit_normalized
    
    @staticmethod
    def create_employment_stability(X: pd.DataFrame) -> pd.Series:
        """
        Create employment stability indicator.
        
        Based on years of employment. More stable employment suggests
        lower fraud risk.
        
        Args:
            X: DataFrame with 'employment_years' column
            
        Returns:
            Series with stability scores (0-1 normalized)
        """
        # Normalize employment years (cap at 30 years)
        stability = (X['employment_years'] / 30).clip(0, 1)
        return stability
    
    @staticmethod
    def create_age_credit_interaction(X: pd.DataFrame) -> pd.Series:
        """
        Create interaction feature between age and credit score.
        
        Captures the relationship between customer maturity and
        credit responsibility.
        
        Args:
            X: DataFrame with 'age' and 'credit_score' columns
            
        Returns:
            Series with interaction values
        """
        age_normalized = X['age'] / 100  # Normalize age
        credit_normalized = X['credit_score'] / 850  # Normalize credit score
        return age_normalized * credit_normalized
    
    @staticmethod
    def create_loan_amount_category(X: pd.DataFrame) -> pd.Series:
        """
        Create categorical feature for loan amount.
        
        Categorizes loans into risk tiers based on amount.
        
        Args:
            X: DataFrame with 'loan_amount' column
            
        Returns:
            Series with loan categories (0=low, 1=medium, 2=high, 3=very_high)
        """
        loan_categories = pd.cut(
            X['loan_amount'],
            bins=[0, 100000, 250000, 500000, np.inf],
            labels=[0, 1, 2, 3]
        )
        return loan_categories.astype('Int64').fillna(-1).astype(int)
    
    @staticmethod
    def create_income_category(X: pd.DataFrame) -> pd.Series:
        """
        Create categorical feature for income level.
        
        Categorizes income into risk tiers.
        
        Args:
            X: DataFrame with 'income' column
            
        Returns:
            Series with income categories (0=low, 1=medium, 2=high, 3=very_high)
        """
        income_categories = pd.cut(
            X['income'],
            bins=[0, 50000, 100000, 150000, np.inf],
            labels=[0, 1, 2, 3]
        )
        return income_categories.astype('Int64').fillna(-1).astype(int)
    
    @classmethod
    def engineer_features(cls, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            X: Input DataFrame with raw features
            
        Returns:
            DataFrame with engineered features added
        """
        X_engineered = X.copy()
        
        # Create new features (only if required columns exist)
        if {'income', 'loan_amount'}.issubset(X.columns):
            X_engineered['income_to_loan_ratio'] = cls.create_income_to_loan_ratio(X)
            X_engineered['loan_amount_category'] = cls.create_loan_amount_category(X)
            X_engineered['income_category'] = cls.create_income_category(X)

        if 'credit_score' in X.columns:
            X_engineered['credit_history_score'] = cls.create_credit_history_score(X)

        if 'employment_years' in X.columns:
            X_engineered['employment_stability'] = cls.create_employment_stability(X)

        if {'age', 'credit_score'}.issubset(X.columns):
            X_engineered['age_credit_interaction'] = cls.create_age_credit_interaction(X)
        
        return X_engineered
