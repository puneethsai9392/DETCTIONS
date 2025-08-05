
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
print("✅ Libraries imported successfully!")


def generate_fraud_data(n_samples=5000):
    np.random.seed(42)

    # Generate normal transactions (90%)
    normal_count = int(n_samples * 0.9)
    fraud_count = n_samples - normal_count

    # Normal transactions
    normal_data = {
        'amount': np.random.lognormal(3, 1, normal_count),
        'time_hour': np.random.randint(0, 24, normal_count),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], normal_count),
        'location_risk': np.random.uniform(0, 0.3, normal_count),
        'user_age': np.random.randint(18, 80, normal_count),
        'days_since_last_transaction': np.random.randint(0, 30, normal_count),
        'transaction_velocity': np.random.uniform(0, 5, normal_count),
        'is_weekend': np.random.choice([0, 1], normal_count, p=[0.7, 0.3]),
        'is_fraud': np.zeros(normal_count)
    }

    # Fraudulent transactions
    fraud_data = {
        'amount': np.random.lognormal(5, 1.5, fraud_count),
        'time_hour': np.random.choice([2, 3, 4, 22, 23], fraud_count),
        'merchant_category': np.random.choice(['online', 'atm', 'unknown'], fraud_count),
        'location_risk': np.random.uniform(0.7, 1.0, fraud_count),
        'user_age': np.random.randint(18, 80, fraud_count),
        'days_since_last_transaction': np.random.randint(0, 2, fraud_count),
        'transaction_velocity': np.random.uniform(10, 50, fraud_count),
        'is_weekend': np.random.choice([0, 1], fraud_count, p=[0.4, 0.6]),
        'is_fraud': np.ones(fraud_count)
    }

    # Combine data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True)

    return df
df = generate_fraud_data(5000)
print(f"✅ Dataset created with {len(df)} transactions")
print(f"📊 Fraud percentage: {df['is_fraud'].mean():.2%}")
print(f"📋 Dataset shape: {df.shape}")

# Cell 3: Data Exploration
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['is_fraud'].value_counts().plot(kind='bar')
plt.title('Fraud Distribution')
plt.xlabel('Is Fraud')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
df.boxplot(column='amount', by='is_fraud')
plt.title('Amount by Fraud Status')

plt.subplot(1, 3, 3)
df.groupby('is_fraud')['time_hour'].hist(alpha=0.5, bins=24)
plt.title('Transaction Hours')
plt.xlabel('Hour')
plt.legend(['Normal', 'Fraud'])

plt.tight_layout()
plt.show()

# Cell 4: Data Preprocessing
# Encode categorical variables
label_encoder = LabelEncoder()
df['merchant_category_encoded'] = label_encoder.fit_transform(df['merchant_category'])

# Create additional features
df['amount_log'] = np.log1p(df['amount'])
df['is_night'] = ((df['time_hour'] >= 22) | (df['time_hour'] <= 5)).astype(int)
df['high_risk_location'] = (df['location_risk'] > 0.5).astype(int)
df['high_velocity'] = (df['transaction_velocity'] > 10).astype(int)

print("✅ Data preprocessing completed!")
feature_columns = ['amount_log', 'time_hour', 'merchant_category_encoded',
                   'location_risk', 'user_age', 'days_since_last_transaction',
                   'transaction_velocity', 'is_weekend', 'is_night',
                   'high_risk_location', 'high_velocity']

X = df[feature_columns]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split completed!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"✅ SMOTE applied!")
print(f"Original training fraud rate: {y_train.mean():.2%}")
print(f"Balanced training fraud rate: {y_train_balanced.mean():.2%}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed!")

# Cell 7: Train Models
print("🚀 Training models...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train_balanced)

# Logistic Regression
lr_model = LogisticRegression(random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train_balanced)

# Isolation Forest
iso_model = IsolationForest(contamination=0.1, random_state=42)
iso_model.fit(X_train_scaled)

print("✅ All models trained successfully!")

# Cell 8: Make Predictions and Evaluate
# Make predictions
rf_pred = rf_model.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test_scaled)
iso_pred = iso_model.predict(X_test_scaled)
iso_pred = np.where(iso_pred == -1, 1, 0)  # Convert to fraud labels

# Calculate AUC scores
rf_auc = roc_auc_score(y_test, rf_pred)
lr_auc = roc_auc_score(y_test, lr_pred)
iso_auc = roc_auc_score(y_test, iso_pred)

print("📊 Model Performance:")
print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(f"Isolation Forest AUC: {iso_auc:.4f}")

# Cell 9: Detailed Evaluation
print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)
print(classification_report(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)
print(classification_report(y_test, lr_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# Cell 10: Feature Importance
# Plot feature importance
importance = rf_model.feature_importances_
feature_names = feature_columns
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(importance)), importance[indices])
plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

print("Top 5 Most Important Features:")
for i in range(5):
    print(f"{i + 1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")

# Cell 11: ROC Curves
plt.figure(figsize=(10, 8))

# Random Forest ROC
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')

# Logistic Regression ROC
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')

# Isolation Forest ROC
iso_scores = -iso_model.decision_function(X_test_scaled)
fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_scores)
plt.plot(fpr_iso, tpr_iso, label=f'Isolation Forest (AUC = {iso_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Fraud Detection Models')
plt.legend()
plt.grid(True)
plt.show()

# Cell 12: Test New Transactions
print("\n🔍 Testing fraud detection on new transactions...")

# Create sample new transactions
new_transactions = pd.DataFrame({
    'amount': [50.0, 5000.0, 25.0, 1000.0],
    'time_hour': [14, 3, 10, 23],
    'merchant_category': ['grocery', 'online', 'restaurant', 'atm'],
    'location_risk': [0.1, 0.9, 0.2, 0.8],
    'user_age': [35, 28, 45, 22],
    'days_since_last_transaction': [5, 0, 12, 1],
    'transaction_velocity': [2, 25, 1, 15],
    'is_weekend': [0, 1, 0, 1]
})

# Preprocess new transactions
new_transactions['merchant_category_encoded'] = label_encoder.transform(new_transactions['merchant_category'])
new_transactions['amount_log'] = np.log1p(new_transactions['amount'])
new_transactions['is_night'] = ((new_transactions['time_hour'] >= 22) | (new_transactions['time_hour'] <= 5)).astype(
    int)
new_transactions['high_risk_location'] = (new_transactions['location_risk'] > 0.5).astype(int)
new_transactions['high_velocity'] = (new_transactions['transaction_velocity'] > 10).astype(int)

# Select features and scale
X_new = new_transactions[feature_columns]
X_new_scaled = scaler.transform(X_new)

# Make predictions
rf_new_pred = rf_model.predict(X_new_scaled)
lr_new_pred = lr_model.predict(X_new_scaled)
iso_new_pred = iso_model.predict(X_new_scaled)
iso_new_pred = np.where(iso_new_pred == -1, 1, 0)

# Display results
print("\nFraud Predictions for New Transactions:")
print("-" * 60)
for i in range(len(new_transactions)):
    trans = new_transactions.iloc[i]
    print(f"\nTransaction {i + 1}:")
    print(f"  Amount: ${trans['amount']:.2f}")
    print(f"  Time: {trans['time_hour']}:00")
    print(f"  Merchant: {trans['merchant_category']}")
    print(f"  Location Risk: {trans['location_risk']:.2f}")
    print(f"  Predictions:")
    print(f"    Random Forest: {'🚨 FRAUD' if rf_new_pred[i] == 1 else '✅ LEGITIMATE'}")
    print(f"    Logistic Regression: {'🚨 FRAUD' if lr_new_pred[i] == 1 else '✅ LEGITIMATE'}")
    print(f"    Isolation Forest: {'🚨 FRAUD' if iso_new_pred[i] == 1 else '✅ LEGITIMATE'}")

print("\n✅ Fraud detection analysis completed!")
print("\n💡 Tips for improving the model:")
print("1. Use real transaction data for better accuracy")
print("2. Add more features (device info, IP location, etc.)")
print("3. Implement real-time monitoring")
print("4. Regular model retraining with new data")
print("5. Consider ensemble me+++thods for better performance")