"""Tests for model training and evaluation."""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.config import ModelConfig


class TestModelTrainer:
    """Test suite for ModelTrainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)
        return X, y
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer()
        assert trainer.models == {}
        assert trainer.evaluator is not None
        assert trainer.config is not None
    
    def test_train_logistic_regression(self, sample_data):
        """Test logistic regression training."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_logistic_regression(X, y)
        assert model is not None
        assert 'logistic_regression' in trainer.models
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_train_random_forest(self, sample_data):
        """Test random forest training."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_random_forest(X, y)
        assert model is not None
        assert 'random_forest' in trainer.models
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_train_xgboost(self, sample_data):
        """Test XGBoost training."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_xgboost(X, y)
        assert model is not None
        assert 'xgboost' in trainer.models
    
    def test_train_all_models(self, sample_data):
        """Test training all models."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        models = trainer.train_all_models(X, y)
        assert len(models) == 3
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert 'xgboost' in models
    
    def test_cross_validation(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_logistic_regression(X, y)
        scores = trainer.cross_validate(model, X, y)
        
        assert 'accuracy' in scores
        assert 'precision' in scores
        assert 'recall' in scores
        assert 'f1' in scores
        assert 'roc_auc' in scores
        
        # Each metric should have mean and std
        for metric, values in scores.items():
            assert 'mean' in values
            assert 'std' in values
            assert 'scores' in values
    
    def test_save_and_load_model(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        # Train model
        model = trainer.train_logistic_regression(X, y)
        
        # Save model
        save_path = tmp_path / "test_model.joblib"
        trainer.save_model(model, "test_model", str(save_path))
        
        # Load model
        loaded_model = trainer.load_model(str(save_path))
        assert loaded_model is not None
        
        # Test predictions match
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        assert np.array_equal(original_pred, loaded_pred)
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        # Train random forest (has feature_importances_)
        model = trainer.train_random_forest(X, y)
        
        # Get importance
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = trainer.get_feature_importance(model, feature_names)
        
        assert len(importance_df) == len(feature_names)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert importance_df['importance'].sum() > 0


class TestModelEvaluator:
    """Test suite for ModelEvaluator."""
    
    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
        y_pred_proba = np.array([
            0.1, 0.2, 0.9, 0.4, 0.15, 0.95, 0.7, 0.1, 0.85, 0.92
        ])
        return y_true, y_pred, y_pred_proba
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.average == 'binary'
        assert evaluator.evaluation_results == {}
    
    def test_evaluate(self, predictions):
        """Test evaluation metrics calculation."""
        y_true, y_pred, y_pred_proba = predictions
        evaluator = ModelEvaluator()
        
        results = evaluator.evaluate(y_true, y_pred, y_pred_proba)
        
        # Check required metrics exist
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'roc_auc' in results
        assert 'specificity' in results
        assert 'false_positive_rate' in results
        
        # Check confusion matrix metrics
        assert 'true_negatives' in results
        assert 'false_positives' in results
        assert 'true_positives' in results
        assert 'false_negatives' in results
        
        # Check values are in valid ranges
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['roc_auc'] <= 1
    
    def test_confusion_matrix_values(self, predictions):
        """Test confusion matrix calculation."""
        y_true, y_pred, _ = predictions
        evaluator = ModelEvaluator()
        
        results = evaluator.evaluate(y_true, y_pred)
        
        # Manually calculate expected confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        assert results['true_negatives'] == tn
        assert results['false_positives'] == fp
        assert results['false_negatives'] == fn
        assert results['true_positives'] == tp
    
    def test_summary_report(self, predictions):
        """Test summary report generation."""
        y_true, y_pred, y_pred_proba = predictions
        evaluator = ModelEvaluator()
        
        evaluator.evaluate(y_true, y_pred, y_pred_proba)
        summary = evaluator.get_summary_report()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 7
        assert 'Metric' in summary.columns
        assert 'Value' in summary.columns
        assert 'Interpretation' in summary.columns
    
    def test_evaluation_without_fit_raises_error(self, predictions):
        """Test that summary report raises error without evaluation."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.get_summary_report()
    
    def test_metrics_calculation_accuracy(self, predictions):
        """Test accuracy metric calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(y_true, y_pred)
        
        # Expected: 5 correct out of 6
        expected_accuracy = 5 / 6
        assert abs(results['accuracy'] - expected_accuracy) < 0.01
    
    def test_business_metrics(self, predictions):
        """Test business-focused metrics."""
        y_true, y_pred, y_pred_proba = predictions
        evaluator = ModelEvaluator()
        
        results = evaluator.evaluate(y_true, y_pred, y_pred_proba)
        
        # False positive rate should be low (minimize rejecting legitimate customers)
        assert results['false_positive_rate'] >= 0
        
        # Fraud detection rate should be high (catch actual fraud)
        assert results['fraud_detection_rate'] >= 0


class TestModelIntegration:
    """Integration tests for model training and evaluation."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        # Create slightly imbalanced binary classification
        y = np.random.binomial(1, 0.3, n_samples)  # 30% fraud rate
        
        return X, y
    
    def test_train_evaluate_pipeline(self, sample_dataset):
        """Test complete train and evaluate pipeline."""
        X, y = sample_dataset
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train
        trainer = ModelTrainer()
        model = trainer.train_random_forest(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(y_test, y_pred, y_pred_proba)
        
        # Check results
        assert results['roc_auc'] > 0.5  # Better than random
        assert results['accuracy'] > 0.5
