"""Data loading and validation utilities for loan application dataset."""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """
    Handles loading, validation, and splitting of loan application data.
    
    This class provides functionality to load raw loan application data,
    validate its integrity, and split it into training and testing sets
    with stratification to maintain fraud distribution.
    """
    
    REQUIRED_COLUMNS = {
        'income', 'loan_amount', 'credit_score', 'employment_years', 'fraud_label'
    }

    OPTIONAL_NUMERICAL_FEATURES = ['age', 'dti']
    OPTIONAL_CATEGORICAL_FEATURES = ['education_level', 'marital_status']
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader with path to data file.
        
        Args:
            data_path: Path to CSV file containing loan application data
            
        Raises:
            FileNotFoundError: If data file does not exist
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file and perform validation.
        
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self.data = pd.read_csv(self.data_path)
        except Exception as e:
            raise ValueError(f"Error reading data file: {str(e)}")
        
        self._validate_data()
        return self.data
    
    def _validate_data(self) -> None:
        """
        Validate data integrity and structure.
        
        Raises:
            ValueError: If data validation fails
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        if self.data.empty:
            raise ValueError("Data file is empty")
        
        # Check for required columns
        missing_cols = self.REQUIRED_COLUMNS - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for duplicates
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            print(f"Warning: {duplicate_count} duplicate rows found")
        
        # Validate fraud label is binary
        fraud_values = self.data['fraud_label'].unique()
        if not set(fraud_values).issubset({0, 1}):
            raise ValueError(f"fraud_label must be binary (0 or 1), found: {fraud_values}")
        
        # Validate numerical features are numeric
        numerical_cols = list(self.REQUIRED_COLUMNS - {'fraud_label'}) + [
            col for col in self.OPTIONAL_NUMERICAL_FEATURES if col in self.data.columns
        ]
        for col in numerical_cols:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Column {col} must be numeric")
    
    def get_feature_split(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and target variable separated.
        
        Returns:
            Tuple of (features, target)
        """
        if self.data is None:
            self.load_data()
        
        X = self.data.drop('fraud_label', axis=1)
        y = self.data['fraud_label']
        return X, y
    
    def load_and_split(
        self, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data and split into training and testing sets with stratification.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X, y = self.get_feature_split()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain fraud distribution
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary containing data statistics
        """
        if self.data is None:
            self.load_data()
        
        fraud_count = (self.data['fraud_label'] == 1).sum()
        legitimate_count = (self.data['fraud_label'] == 0).sum()
        fraud_ratio = fraud_count / len(self.data) * 100
        
        return {
            'total_samples': len(self.data),
            'fraud_cases': fraud_count,
            'legitimate_cases': legitimate_count,
            'fraud_percentage': fraud_ratio,
            'features': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict()
        }
