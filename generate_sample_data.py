"""
Generate sample loan application data for testing and demonstration.

Run this script to create a sample dataset for the fraud detection model.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(
    n_samples: int = 2000,
    fraud_rate: float = 0.25,
    random_state: int = 42,
    output_path: str = 'data/raw/loan_applications.csv'
) -> pd.DataFrame:
    """
    Generate synthetic loan application data.
    
    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent cases (0-1)
        random_state: Random seed for reproducibility
        output_path: Path to save generated CSV
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(random_state)
    
    # Generate features
    income = np.random.uniform(40000, 150000, n_samples)
    loan_amount = np.random.uniform(100000, 400000, n_samples)
    credit_score = np.random.normal(720, 80, n_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    employment_years = np.random.randint(0, 30, n_samples)
    age = np.random.randint(25, 65, n_samples)
    education_level = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n_samples
    )
    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced'],
        n_samples
    )
    
    # Generate fraud labels
    fraud_label = np.random.binomial(1, fraud_rate, n_samples)
    
    # Adjust features for fraudulent cases to make them detectable
    fraud_mask = fraud_label == 1
    
    # Fraudsters tend to have lower income-to-loan ratio
    income[fraud_mask] *= 0.7
    # Lower credit scores
    credit_score[fraud_mask] -= 50
    # Less employment stability
    employment_years[fraud_mask] *= 0.5
    # Slightly younger or older
    age[fraud_mask] -= 5
    
    # Create DataFrame
    df = pd.DataFrame({
        'income': income,
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'age': age,
        'education_level': education_level,
        'marital_status': marital_status,
        'fraud_label': fraud_label.astype(int)
    })
    
    # Ensure valid ranges
    df['credit_score'] = df['credit_score'].clip(300, 850).astype(int)
    df['employment_years'] = df['employment_years'].clip(0, None).astype(int)
    df['age'] = df['age'].clip(18, 100).astype(int)
    df['income'] = df['income'].round(2)
    df['loan_amount'] = df['loan_amount'].round(2)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample data generated successfully!")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Fraud rate: {fraud_label.mean()*100:.1f}%")
    print(f"  Saved to: {output_path}")
    
    return df


def display_sample_statistics(df: pd.DataFrame) -> None:
    """
    Display statistics about the generated data.
    
    Args:
        df: Generated DataFrame
    """
    print("\n" + "="*60)
    print("SAMPLE DATA STATISTICS")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape[0]} samples × {df.shape[1]} features")
    
    print("\nFeature Overview:")
    print(f"  Income range: ${df['income'].min():,.0f} - ${df['income'].max():,.0f}")
    print(f"  Loan amount range: ${df['loan_amount'].min():,.0f} - ${df['loan_amount'].max():,.0f}")
    print(f"  Credit score range: {df['credit_score'].min()} - {df['credit_score'].max()}")
    print(f"  Employment years range: {df['employment_years'].min()} - {df['employment_years'].max()}")
    print(f"  Age range: {df['age'].min()} - {df['age'].max()}")
    
    print("\nCategorical Features:")
    print(f"  Education levels: {df['education_level'].nunique()}")
    print(f"    {df['education_level'].value_counts().to_dict()}")
    print(f"  Marital status: {df['marital_status'].nunique()}")
    print(f"    {df['marital_status'].value_counts().to_dict()}")
    
    print("\nFraud Distribution:")
    fraud_counts = df['fraud_label'].value_counts()
    print(f"  Legitimate: {fraud_counts[0]} ({fraud_counts[0]/len(df)*100:.1f}%)")
    print(f"  Fraudulent: {fraud_counts[1]} ({fraud_counts[1]/len(df)*100:.1f}%)")
    
    print("\nMissing Values: None (synthetic data)")
    
    print("\nFirst 5 Rows:")
    print(df.head().to_string(index=False))


if __name__ == '__main__':
    # Generate sample data
    df = generate_sample_data(
        n_samples=2000,
        fraud_rate=0.25,
        random_state=42,
        output_path='data/raw/loan_applications.csv'
    )
    
    # Display statistics
    display_sample_statistics(df)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. You can now use this data with the fraud detection models:
   
   from src.data_loader import DataLoader
   loader = DataLoader('data/raw/loan_applications.csv')
   X_train, X_test, y_train, y_test = loader.load_and_split()

2. Start with the Jupyter notebooks:
   
   jupyter notebook notebooks/01_eda.ipynb

3. Or run unit tests:
   
   pytest tests/ -v

Note: This is synthetic data for demonstration purposes.
      Replace with real data for production use.
    """)
