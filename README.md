# Fraud Detection Using Machine Learning

This project implements an end-to-end machine learning pipeline to detect fraudulent transactions using synthetic data. It covers data generation, preprocessing, handling class imbalance, model training, evaluation, and testing on new transactions.

## Features

- Synthetic dataset simulating normal and fraudulent transaction patterns
- Data exploration and visualization for fraud insights
- Feature engineering including categorical encoding and risk flag creation
- Class imbalance handling using SMOTE oversampling
- Training and evaluation of Random Forest, Logistic Regression, and Isolation Forest models
- Performance metrics including ROC-AUC, classification reports, and confusion matrices
- Visualization of feature importance and ROC curves
- Demonstration of fraud prediction on new unseen transactions

## Results

- Achieved strong classification performance (ROC-AUC scores) on test data
- Identified key features contributing to fraud detection
- Provided visualization for model interpretability and decision making

## Future Work

- Incorporate real-world transaction data for improved accuracy
- Add advanced features such as device fingerprints, IP location
- Develop real-time fraud alert systems
- Explore deep learning and ensemble methods for enhanced detection

## Author

**Puneeth Sai**  
[GitHub Profile](https://github.com/puneethsai)  
Email: puneethsai632@gmail.com

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Usage

1. Generate synthetic fraud dataset:  
   ```python
   generate_fraud_data()
   ```
2. Perform exploratory data analysis and preprocess features
3. Apply SMOTE to balance the training data
4. Train ML models and evaluate on test data
5. Use trained models to predict fraud on new transactions

---

Feel free to fork, contribute, or raise issues for improvements!
