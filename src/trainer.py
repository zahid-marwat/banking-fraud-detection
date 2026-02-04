"""Model training pipeline for fraud detection."""

from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

from src.config import CONFIG
from src.evaluator import ModelEvaluator


class ModelTrainer:
    """
    Training pipeline for fraud detection models.
    
    Handles model instantiation, training with class imbalance handling,
    cross-validation, and model persistence.
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object (uses global CONFIG if None)
        """
        self.config = config or CONFIG
        self.models: Dict[str, Any] = {}
        self.evaluator = ModelEvaluator()
        self.scaler = None
        self.smote = None
    
    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Balanced features and labels
        """
        smote = SMOTE(
            sampling_strategy=self.config.SMOTE_RATIO,
            random_state=self.config.RANDOM_STATE
        )
        X_smote, y_smote = smote.fit_resample(X, y)
        return X_smote, y_smote
    
    def _instantiate_logistic_regression(self) -> LogisticRegression:
        """
        Instantiate Logistic Regression model.
        
        Returns:
            LogisticRegression instance
        """
        return LogisticRegression(**self.config.LOGISTIC_REGRESSION_PARAMS)
    
    def _instantiate_random_forest(self) -> RandomForestClassifier:
        """
        Instantiate Random Forest model.
        
        Returns:
            RandomForestClassifier instance
        """
        return RandomForestClassifier(**self.config.RANDOM_FOREST_PARAMS)
    
    def _instantiate_xgboost(self) -> XGBClassifier:
        """
        Instantiate XGBoost model.
        
        Returns:
            XGBClassifier instance
        """
        return XGBClassifier(**self.config.XGBOOST_PARAMS)
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        X_processed = X_train.copy()
        y_processed = y_train.copy()
        
        # Apply SMOTE if configured
        if self.config.USE_SMOTE:
            X_processed, y_processed = self._apply_smote(X_processed, y_processed)
        
        model = self._instantiate_logistic_regression()
        model.fit(X_processed, y_processed)
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        X_processed = X_train.copy()
        y_processed = y_train.copy()
        
        # Apply SMOTE if configured
        if self.config.USE_SMOTE:
            X_processed, y_processed = self._apply_smote(X_processed, y_processed)
        
        model = self._instantiate_random_forest()
        model.fit(X_processed, y_processed)
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        X_processed = X_train.copy()
        y_processed = y_train.copy()
        
        # Apply SMOTE if configured
        if self.config.USE_SMOTE:
            X_processed, y_processed = self._apply_smote(X_processed, y_processed)
        
        model = self._instantiate_xgboost()
        model.fit(X_processed, y_processed, verbose=False)
        
        self.models['xgboost'] = model
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        print("Training Logistic Regression...")
        self.train_logistic_regression(X_train, y_train)
        
        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train)
        
        print("Training XGBoost...")
        self.train_xgboost(X_train, y_train)
        
        return self.models
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> VotingClassifier:
        """
        Train ensemble model combining multiple classifiers.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained ensemble model
        """
        # Ensure individual models are trained
        if 'logistic_regression' not in self.models:
            self.train_logistic_regression(X_train, y_train)
        if 'random_forest' not in self.models:
            self.train_random_forest(X_train, y_train)
        if 'xgboost' not in self.models:
            self.train_xgboost(X_train, y_train)
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ],
            voting='soft'
        )
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform cross-validation on model.
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            
        Returns:
            Cross-validation scores
        """
        cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        
        scores = {}
        for metric in self.config.METRICS:
            score = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=metric
            )
            scores[metric] = {
                'mean': score.mean(),
                'std': score.std(),
                'scores': score
            }
        
        return scores
    
    def save_model(self, model: Any, model_name: str, path: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model: Model to save
            model_name: Name for the model
            path: Custom path to save (uses default if None)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            model_dir = Path(self.config.MODEL_DIR)
            model_dir.mkdir(exist_ok=True)
            path = model_dir / f"{model_name}.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, path)
        print(f"Model saved to: {path}")
        return str(path)
    
    def load_model(self, path: str) -> Any:
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(path)
        print(f"Model loaded from: {path}")
        return model
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model: Trained model
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importances
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {type(model).__name__} does not have feature_importances_")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
