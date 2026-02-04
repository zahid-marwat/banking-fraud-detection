"""
BANKING FRAUD DETECTION ML PROJECT - SETUP COMPLETE

This document provides quick reference information about the project structure,
setup instructions, and usage examples.
"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_STRUCTURE = """
banking-fraud-detection/
├── data/
│   ├── raw/                    # Place your CSV files here
│   └── processed/              # Generated during preprocessing
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing & feature engineering
│   └── 03_modeling.ipynb       # Model training & comparison
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── data_loader.py          # Data loading & validation
│   ├── preprocessor.py         # Preprocessing & feature engineering
│   ├── trainer.py              # Model training pipeline
│   ├── evaluator.py            # Model evaluation metrics
│   └── predict.py              # Prediction on new data
├── models/                     # Trained model checkpoints
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py    # Data preprocessing tests
│   └── test_models.py          # Model functionality tests
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
"""

# ============================================================================
# QUICK START GUIDE
# ============================================================================

QUICK_START = """
1. INSTALL DEPENDENCIES:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. PREPARE YOUR DATA:
   - Place CSV file in: data/raw/loan_applications.csv
   - Required columns: income, loan_amount, credit_score, employment_years,
                       age, education_level, marital_status, fraud_label
   - fraud_label should be binary (0=legitimate, 1=fraudulent)

3. RUN JUPYTER NOTEBOOKS:
   jupyter notebook notebooks/
   
   - Start with 01_eda.ipynb for data exploration
   - Then 02_preprocessing.ipynb for data preparation
   - Finally 03_modeling.ipynb for model training

4. RUN UNIT TESTS:
   pytest tests/ -v
   pytest tests/ --cov=src  # With coverage report

5. TRAIN MODELS PROGRAMMATICALLY:
   from src.trainer import ModelTrainer
   from src.data_loader import DataLoader
   
   loader = DataLoader('data/raw/loan_applications.csv')
   X_train, X_test, y_train, y_test = loader.load_and_split()
   
   trainer = ModelTrainer()
   models = trainer.train_all_models(X_train, y_train)

6. MAKE PREDICTIONS:
   from src.predict import FraudPredictor
   
   predictor = FraudPredictor('models/best_model.joblib')
   new_app = {'income': 75000, 'loan_amount': 250000, ...}
   result = predictor.predict_single(new_app)
"""

# ============================================================================
# KEY FEATURES
# ============================================================================

KEY_FEATURES = """
✓ MULTIPLE MODELS:
  - Logistic Regression (interpretability)
  - Random Forest (non-linear patterns)
  - XGBoost (gradient boosting)
  - Ensemble methods (robustness)

✓ CLASS IMBALANCE HANDLING:
  - SMOTE (Synthetic Minority Oversampling)
  - Class weights in model training
  - Stratified cross-validation
  - Appropriate evaluation metrics

✓ FEATURE ENGINEERING:
  - Income-to-loan ratio (debt burden)
  - Credit history score (payment reliability)
  - Employment stability indicator
  - Age-credit score interaction
  - Categorical discretization

✓ COMPREHENSIVE EVALUATION:
  - Precision, Recall, F1-Score
  - ROC-AUC curve
  - Precision-Recall curve
  - Confusion matrix analysis
  - Business-focused metrics

✓ PRODUCTION-READY CODE:
  - Type hints on all functions
  - Comprehensive docstrings
  - Error handling
  - Unit tests with 20+ test cases
  - PEP 8 compliant
"""

# ============================================================================
# IMPORTANT CONFIGURATIONS
# ============================================================================

IMPORTANT_CONFIGS = """
src/config.py - Key parameters:

TEST_SIZE = 0.2                    # 80-20 train-test split
USE_SMOTE = True                   # Handle class imbalance
USE_CLASS_WEIGHTS = True           # Additional imbalance handling
CV_FOLDS = 5                       # Cross-validation folds
PRIMARY_METRIC = 'roc_auc'         # Best for fraud detection
STRATIFIED_CV = True               # Maintain fraud distribution

MODEL PARAMETERS:
- Logistic Regression: max_iter=1000, class_weight='balanced'
- Random Forest: n_estimators=100, max_depth=15, class_weight='balanced'
- XGBoost: n_estimators=100, max_depth=5, learning_rate=0.1

Adjust these parameters in config.py for your specific needs
"""

# ============================================================================
# DATA REQUIREMENTS
# ============================================================================

DATA_REQUIREMENTS = """
Expected CSV Format:

Column Name            Type        Description
income                 numeric     Annual income ($)
loan_amount            numeric     Requested loan amount ($)
credit_score           numeric     Credit score (300-850)
employment_years       numeric     Years at current job
age                    numeric     Applicant age
education_level        categorical High School, Bachelor, Master, PhD
marital_status         categorical Single, Married, Divorced
fraud_label            binary      0=legitimate, 1=fraudulent

Example Row:
income=75000, loan_amount=250000, credit_score=720, employment_years=5,
age=45, education_level=Bachelor, marital_status=Married, fraud_label=0

Data Quality Checks:
✓ No NULL values in fraud_label
✓ Numerical features are actually numeric
✓ fraud_label contains only 0 and 1
✓ No duplicate records (checked and reported)
"""

# ============================================================================
# BUSINESS METRICS EXPLAINED
# ============================================================================

BUSINESS_METRICS = """
METRICS FOR FRAUD DETECTION:

1. PRECISION (False Positive Rate):
   → What % of flagged applications are actually fraud?
   → High precision = Few legitimate customers rejected
   → Goal: Minimize customer frustration from false rejections

2. RECALL (Fraud Detection Rate):
   → What % of actual fraud cases do we catch?
   → High recall = Catch more fraudsters
   → Goal: Minimize fraud losses

3. F1-SCORE:
   → Balanced measure of precision and recall
   → Good when you need balance between both metrics

4. ROC-AUC:
   → Discriminative ability across all thresholds
   → Robust to class imbalance (best metric for fraud)
   → Higher AUC = Better model

5. SPECIFICITY:
   → Ability to identify legitimate applications
   → True Negative Rate
   → Important for customer experience

THRESHOLD TUNING:
Default threshold = 0.5 (50% probability fraud)

- LOWER threshold (0.3) → More fraud detection, more false positives
  Use when: Fraud loss >> customer frustration

- HIGHER threshold (0.7) → Fewer false positives, miss some fraud
  Use when: Customer retention >> fraud prevention
"""

# ============================================================================
# FEATURE ENGINEERING DETAILS
# ============================================================================

FEATURE_ENGINEERING = """
ENGINEERED FEATURES:

1. income_to_loan_ratio:
   Formula: income / loan_amount
   Interpretation: Debt burden indicator
   Lower ratio = Higher risk

2. credit_history_score:
   Formula: (credit_score - 300) / 550 (normalized to 0-1)
   Interpretation: Payment reliability (0=poor, 1=excellent)

3. employment_stability:
   Formula: employment_years / 30 (normalized to 0-1)
   Interpretation: Job continuity indicator

4. age_credit_interaction:
   Formula: (age/100) * (credit_score/850)
   Interpretation: Maturity × creditworthiness

5. loan_amount_category:
   Bins: [0-100k, 100k-250k, 250k-500k, 500k+]
   Interpretation: Loan risk tier

6. income_category:
   Bins: [0-50k, 50k-100k, 100k-150k, 150k+]
   Interpretation: Income tier

These features help models learn fraud patterns more effectively
"""

# ============================================================================
# COMMON ISSUES & SOLUTIONS
# ============================================================================

TROUBLESHOOTING = """
ISSUE: FileNotFoundError: Data file not found
SOLUTION: Place your CSV file at data/raw/loan_applications.csv

ISSUE: Low model performance (ROC-AUC < 0.7)
SOLUTIONS:
- Check data quality and missing values
- Verify fraud_label is properly balanced (not 95%+ one class)
- Increase model complexity (more estimators, deeper trees)
- Tune hyperparameters using GridSearchCV
- Create more informative features

ISSUE: Model overfitting (Train ROC-AUC >> Test ROC-AUC)
SOLUTIONS:
- Increase regularization (lower C in Logistic Regression)
- Reduce tree depth (max_depth parameter)
- Increase cross-validation folds
- Use more training data
- Remove redundant features

ISSUE: Class imbalance causing poor fraud detection
SOLUTIONS:
- SMOTE is already enabled by default
- Increase class_weight for minority class
- Lower prediction threshold for fraud class
- Use evaluation metrics that don't penalize imbalance (ROC-AUC, F1)

ISSUE: Tests failing on Windows
SOLUTION: Ensure paths use backslashes or raw strings
           Example: r'path\to\file' or 'path/to/file'

ISSUE: Memory error with large datasets
SOLUTION:
- Process data in batches
- Reduce sample size for initial testing
- Use models with lower memory footprint
- Increase available RAM or use cloud computing
"""

# ============================================================================
# NEXT STEPS FOR PRODUCTION
# ============================================================================

PRODUCTION_CHECKLIST = """
□ Validate model performance on hold-out test set
□ Implement model monitoring and logging
□ Set up automated retraining pipeline
□ Create data quality validation checks
□ Implement prediction API (Flask, FastAPI, etc.)
□ Set up A/B testing for threshold changes
□ Document decision rules for model updates
□ Create audit trail for all predictions
□ Implement explainability (SHAP, LIME)
□ Set up performance dashboards
□ Create incident response procedures
□ Establish model governance framework
□ Regular model performance reviews (monthly/quarterly)
□ Update models as fraud patterns evolve
□ Maintain feature importance tracking
□ Document all configuration changes
"""

# ============================================================================
# RESOURCES & DOCUMENTATION
# ============================================================================

RESOURCES = """
KEY FILES REFERENCE:

Source Code Organization:
- src/data_loader.py: Load and validate loan application data
- src/preprocessor.py: Data cleaning and feature engineering
- src/trainer.py: Model training with multiple algorithms
- src/evaluator.py: Comprehensive evaluation metrics
- src/predict.py: Make predictions on new applications
- src/config.py: Centralized configuration

Tests:
- tests/test_preprocessor.py: Data pipeline unit tests
- tests/test_models.py: Model training and evaluation tests

Notebooks (Start here!):
1. 01_eda.ipynb: Understand your data
2. 02_preprocessing.ipynb: Prepare data for modeling
3. 03_modeling.ipynb: Train and compare models

Documentation:
- README.md: Complete project overview
- This file: Quick reference and troubleshooting

External Resources:
- scikit-learn docs: https://scikit-learn.org/
- XGBoost guide: https://xgboost.readthedocs.io/
- Imbalanced-learn: https://imbalanced-learn.org/
- SHAP for interpretability: https://shap.readthedocs.io/
"""

# ============================================================================
# PRINT ALL SECTIONS
# ============================================================================

if __name__ == '__main__':
    sections = [
        ("PROJECT STRUCTURE", PROJECT_STRUCTURE),
        ("QUICK START GUIDE", QUICK_START),
        ("KEY FEATURES", KEY_FEATURES),
        ("IMPORTANT CONFIGURATIONS", IMPORTANT_CONFIGS),
        ("DATA REQUIREMENTS", DATA_REQUIREMENTS),
        ("BUSINESS METRICS EXPLAINED", BUSINESS_METRICS),
        ("FEATURE ENGINEERING DETAILS", FEATURE_ENGINEERING),
        ("COMMON ISSUES & SOLUTIONS", TROUBLESHOOTING),
        ("PRODUCTION CHECKLIST", PRODUCTION_CHECKLIST),
        ("RESOURCES & DOCUMENTATION", RESOURCES)
    ]
    
    for title, content in sections:
        print("\n" + "="*70)
        print(f"{title:^70}")
        print("="*70)
        print(content)
    
    print("\n" + "="*70)
    print("✓ PROJECT SETUP COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Place data in: data/raw/loan_applications.csv")
    print("3. Start with: jupyter notebook notebooks/01_eda.ipynb")
    print("4. Run tests: pytest tests/ -v")
    print("\nFor questions, refer to README.md or inline documentation in source files.")
