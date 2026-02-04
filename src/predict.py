"""Prediction module for fraud detection model."""

from typing import Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.preprocessor import DataPreprocessor, FeatureEngineer


class FraudPredictor:
    """
    Makes predictions on new loan applications for fraud detection.
    
    Handles preprocessing, feature engineering, and prediction
    with proper error handling and result formatting.
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model file
            preprocessor: Fitted DataPreprocessor (optional)
            
        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.preprocessor = preprocessor or DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
    
    def predict_single(
        self,
        application: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make prediction on single loan application.
        
        Args:
            application: Dictionary with application features
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Convert to DataFrame
        app_df = pd.DataFrame([application])
        
        # Engineer features
        app_engineered = self.feature_engineer.engineer_features(app_df)
        
        # Preprocess
        app_processed = self.preprocessor.transform(app_engineered)
        
        # Make prediction
        prediction = self.model.predict(app_processed)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(app_processed)[0]
            fraud_probability = float(probability[1])
        else:
            fraud_probability = None
        
        return {
            'prediction': int(prediction),
            'is_fraud': bool(prediction == 1),
            'fraud_probability': fraud_probability,
            'risk_level': self._get_risk_level(fraud_probability)
        }
    
    def predict_batch(
        self,
        applications: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Make predictions on multiple loan applications.
        
        Args:
            applications: List of application dictionaries
            
        Returns:
            DataFrame with predictions for all applications
        """
        # Convert to DataFrame
        apps_df = pd.DataFrame(applications)
        
        # Engineer features
        apps_engineered = self.feature_engineer.engineer_features(apps_df)
        
        # Preprocess
        apps_processed = self.preprocessor.transform(apps_engineered)
        
        # Make predictions
        predictions = self.model.predict(apps_processed)
        
        # Get probabilities if available
        results_data = {
            'prediction': predictions,
            'is_fraud': predictions == 1
        }
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(apps_processed)
            results_data['fraud_probability'] = probabilities[:, 1]
            results_data['risk_level'] = [
                self._get_risk_level(prob) for prob in probabilities[:, 1]
            ]
        
        results_df = pd.DataFrame(results_data)
        return results_df
    
    def predict_proba_single(
        self,
        application: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get prediction probabilities for single application.
        
        Args:
            application: Dictionary with application features
            
        Returns:
            Dictionary with class probabilities
            
        Raises:
            ValueError: If model doesn't support probability predictions
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Convert to DataFrame
        app_df = pd.DataFrame([application])
        
        # Engineer features
        app_engineered = self.feature_engineer.engineer_features(app_df)
        
        # Preprocess
        app_processed = self.preprocessor.transform(app_engineered)
        
        # Get probabilities
        probabilities = self.model.predict_proba(app_processed)[0]
        
        return {
            'legitimate_probability': float(probabilities[0]),
            'fraud_probability': float(probabilities[1])
        }
    
    @staticmethod
    def _get_risk_level(fraud_probability: Optional[float]) -> str:
        """
        Determine risk level from fraud probability.
        
        Args:
            fraud_probability: Predicted fraud probability
            
        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        if fraud_probability is None:
            return 'unknown'
        
        if fraud_probability < 0.2:
            return 'low'
        elif fraud_probability < 0.5:
            return 'medium'
        elif fraud_probability < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': type(self.model).__name__,
            'has_predict_proba': hasattr(self.model, 'predict_proba'),
            'supports_feature_importance': hasattr(self.model, 'feature_importances_')
        }


class PredictionBatch:
    """
    Handles batch prediction operations with result aggregation.
    """
    
    def __init__(self, predictor: FraudPredictor):
        """
        Initialize batch processor.
        
        Args:
            predictor: FraudPredictor instance
        """
        self.predictor = predictor
    
    def predict_and_filter(
        self,
        applications: List[Dict[str, Any]],
        fraud_threshold: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Predict and filter applications by fraud risk.
        
        Args:
            applications: List of applications
            fraud_threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary with DataFrames for legitimate and fraud applications
        """
        predictions = self.predictor.predict_batch(applications)
        
        # Add original application data
        apps_df = pd.DataFrame(applications)
        predictions = pd.concat([apps_df, predictions], axis=1)
        
        # Filter by threshold
        legitimate = predictions[
            predictions.get('fraud_probability', predictions['prediction'] == 0) < fraud_threshold
        ]
        fraud = predictions[
            predictions.get('fraud_probability', predictions['prediction'] == 1) >= fraud_threshold
        ]
        
        return {
            'legitimate': legitimate,
            'fraud': fraud,
            'summary': {
                'total_applications': len(predictions),
                'flagged_as_fraud': len(fraud),
                'approved_legitimate': len(legitimate),
                'fraud_rate': len(fraud) / len(predictions) if len(predictions) > 0 else 0
            }
        }
