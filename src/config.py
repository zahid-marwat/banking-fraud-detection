"""Configuration settings for fraud detection models."""

from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    
    # Paths
    DATA_DIR: Path = Path('data')
    MODEL_DIR: Path = Path('models')
    NOTEBOOK_DIR: Path = Path('notebooks')
    
    # Data parameters
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Class imbalance handling
    USE_SMOTE: bool = True
    SMOTE_RATIO: float = 0.8  # Balance minority class to this ratio
    USE_CLASS_WEIGHTS: bool = True
    
    # Model hyperparameters
    LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = None
    RANDOM_FOREST_PARAMS: Dict[str, Any] = None
    XGBOOST_PARAMS: Dict[str, Any] = None
    
    # Evaluation metrics
    PRIMARY_METRIC: str = 'roc_auc'  # Focus metric for fraud detection
    METRICS: list = None
    
    # Cross-validation
    CV_FOLDS: int = 5
    STRATIFIED_CV: bool = True
    
    # Feature engineering
    ENGINEER_FEATURES: bool = True
    SCALE_FEATURES: bool = True
    
    def __post_init__(self):
        """Initialize default hyperparameters if not provided."""
        if self.LOGISTIC_REGRESSION_PARAMS is None:
            self.LOGISTIC_REGRESSION_PARAMS = {
                'max_iter': 1000,
                'random_state': self.RANDOM_STATE,
                'class_weight': 'balanced' if self.USE_CLASS_WEIGHTS else None
            }
        
        if self.RANDOM_FOREST_PARAMS is None:
            self.RANDOM_FOREST_PARAMS = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.RANDOM_STATE,
                'class_weight': 'balanced' if self.USE_CLASS_WEIGHTS else None,
                'n_jobs': -1
            }
        
        if self.XGBOOST_PARAMS is None:
            self.XGBOOST_PARAMS = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.RANDOM_STATE,
                'scale_pos_weight': 1 if not self.USE_CLASS_WEIGHTS else None
            }
        
        if self.METRICS is None:
            self.METRICS = [
                'accuracy',
                'precision',
                'recall',
                'f1',
                'roc_auc'
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'data_dir': str(self.DATA_DIR),
            'model_dir': str(self.MODEL_DIR),
            'test_size': self.TEST_SIZE,
            'use_smote': self.USE_SMOTE,
            'use_class_weights': self.USE_CLASS_WEIGHTS,
            'cv_folds': self.CV_FOLDS,
            'primary_metric': self.PRIMARY_METRIC,
            'metrics': self.METRICS
        }


# Global configuration instance
CONFIG = ModelConfig()
