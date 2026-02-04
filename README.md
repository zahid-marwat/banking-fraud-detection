# Banking Fraud Detection ML Project

## Overview

A comprehensive machine learning solution for detecting fraudulent loan applications in banking systems. This project implements multiple classification algorithms with advanced preprocessing techniques and class imbalance handling to minimize false positives while maximizing fraud detection accuracy.

## Key Features

- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, and Ensemble methods
- **Class Imbalance Handling**: SMOTE and class weights implementation
- **Feature Engineering**: Income-to-loan ratio, credit history scoring, employment stability indicators
- **Comprehensive Evaluation**: Precision, recall, F1-score, ROC-AUC metrics with focus on business impact
- **Production Ready**: Type hints, docstrings, error handling, and unit tests
- **Interactive Analysis**: Jupyter notebooks for EDA, preprocessing, and modeling

## Project Structure

```
banking-fraud-detection/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data Cleaning & Feature Engineering
│   └── 03_modeling.ipynb       # Model Training & Comparison
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading & validation
│   │   └── preprocessor.py     # Data preprocessing & feature engineering
│   └── models/
│       ├── __init__.py
│       ├── trainer.py          # Model training pipeline
│       ├── evaluator.py        # Model evaluation & metrics
│       └── predict.py          # Prediction on new data
├── models/                     # Trained model checkpoints
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py    # Data preprocessing tests
│   └── test_models.py          # Model functionality tests
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Development dependencies
└── README.md                   # Project documentation
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository
```bash
git clone <repository-url>
cd banking-fraud-detection
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. For development (optional)
```bash
pip install -r requirements-dev.txt
```

## Usage

### Running Analysis Notebooks
```bash
jupyter notebook notebooks/
```

1. **01_eda.ipynb** - Explore fraud patterns and data distributions
2. **02_preprocessing.ipynb** - Experiment with cleaning and feature engineering
3. **03_modeling.ipynb** - Train and compare models

### Training Models
```python
from src.models.trainer import ModelTrainer
from src.data.data_loader import DataLoader

# Load data
loader = DataLoader('data/raw/loan_applications.csv')
X_train, X_test, y_train, y_test = loader.load_and_split()

# Train models
trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)

# Evaluate
evaluator = trainer.evaluate_models(models, X_test, y_test)
```

### Making Predictions
```python
from src.models.predict import FraudPredictor

predictor = FraudPredictor('models/best_model.joblib')
new_applications = [
    {
        'income': 75000,
        'loan_amount': 250000,
        'credit_score': 720,
        'employment_years': 5,
        # ... other features
    }
]
predictions = predictor.predict(new_applications)
```

## Data Format

Expected CSV with loan application features:
```
income, loan_amount, credit_score, employment_years, 
age, education_level, marital_status, fraud_label
```

## Model Performance

Models are evaluated on:
- **Precision**: Minimize false fraud accusations
- **Recall**: Maximize actual fraud detection
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall classification performance
- **Confusion Matrix**: Detailed prediction breakdown

## Feature Engineering

- **Income-to-Loan Ratio**: Debt burden indicator
- **Credit History Score**: Payment reliability
- **Employment Stability**: Job continuity indicator
- **Age-Credit Score Interaction**: Risk profile
- **Categorical Encoding**: One-hot and target encoding

## Handling Class Imbalance

- SMOTE (Synthetic Minority Oversampling)
- Class weights in model training
- Stratified cross-validation
- Custom evaluation metrics for imbalanced data

## Testing

Run unit tests:
```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src
```

## Development Standards

- **Type Hints**: Full type annotations on functions
- **Docstrings**: Google-style docstrings for all functions
- **PEP 8**: Adherence to Python style guide
- **Error Handling**: Comprehensive exception handling
- **Code Quality**: Enforced through linting and formatting

## Best Practices

1. **Minimize False Positives**: Legitimate customers should rarely be incorrectly flagged
2. **Business Impact**: Focus on metrics that align with business goals
3. **Model Interpretability**: Understand which features drive fraud predictions
4. **Regular Retraining**: Update models as fraud patterns evolve
5. **Data Privacy**: Handle sensitive financial data securely

## Contributing

1. Follow PEP 8 style guide
2. Add tests for new features
3. Update documentation
4. Run full test suite before submitting PR

## License

See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact the development team.
