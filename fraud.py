import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#c9d1d9',
    'ytick.color':      '#c9d1d9',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
COLORS = ['#58a6ff', '#f85149', '#3fb950', '#d29922', '#bc8cff', '#ff7b72']
print("✅ Libraries imported successfully!")


# ══════════════════════════════════════════════════════════════
# STEP 1 — INGEST
# ══════════════════════════════════════════════════════════════
def ingest(n_samples=100_000):
    """Generate synthetic transaction dataset with 100,000 records."""
    np.random.seed(42)
    normal_count = int(n_samples * 0.90)
    fraud_count  = n_samples - normal_count

    normal = {
        'amount':                    np.random.lognormal(3, 1,   normal_count),
        'time_hour':                 np.random.randint(0, 24,    normal_count),
        'merchant_category':         np.random.choice(['grocery','gas','restaurant','retail'], normal_count),
        'location_risk':             np.random.uniform(0, 0.3,   normal_count),
        'user_age':                  np.random.randint(18, 80,   normal_count),
        'days_since_last_transaction': np.random.randint(0, 30,  normal_count),
        'transaction_velocity':      np.random.uniform(0, 5,     normal_count),
        'is_weekend':                np.random.choice([0,1],     normal_count, p=[0.7,0.3]),
        'is_fraud':                  np.zeros(normal_count),
    }
    fraud = {
        'amount':                    np.random.lognormal(5, 1.5, fraud_count),
        'time_hour':                 np.random.choice([2,3,4,22,23], fraud_count),
        'merchant_category':         np.random.choice(['online','atm','unknown'], fraud_count),
        'location_risk':             np.random.uniform(0.7, 1.0, fraud_count),
        'user_age':                  np.random.randint(18, 80,   fraud_count),
        'days_since_last_transaction': np.random.randint(0, 2,   fraud_count),
        'transaction_velocity':      np.random.uniform(10, 50,   fraud_count),
        'is_weekend':                np.random.choice([0,1],     fraud_count, p=[0.4,0.6]),
        'is_fraud':                  np.ones(fraud_count),
    }
    combined = {k: np.concatenate([normal[k], fraud[k]]) for k in normal}
    df = pd.DataFrame(combined).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

df = ingest(100_000)
print(f"✅ [INGEST] Dataset: {len(df):,} transactions | Fraud rate: {df['is_fraud'].mean():.2%}")


# ═════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS
# ═════════════════════════════════════════════════════════════
def preprocess(df):
    """Feature engineering and encoding."""
    le = LabelEncoder()
    df['merchant_category_encoded'] = le.fit_transform(df['merchant_category'])
    df['amount_log']          = np.log1p(df['amount'])
    df['is_night']            = ((df['time_hour'] >= 22) | (df['time_hour'] <= 5)).astype(int)
    df['high_risk_location']  = (df['location_risk'] > 0.5).astype(int)
    df['high_velocity']       = (df['transaction_velocity'] > 10).astype(int)
    df['amount_x_velocity']   = df['amount_log'] * df['transaction_velocity']
    df['risk_x_velocity']     = df['location_risk'] * df['transaction_velocity']
    return df, le

df, label_encoder = preprocess(df)
FEATURES = ['amount_log','time_hour','merchant_category_encoded','location_risk',
            'user_age','days_since_last_transaction','transaction_velocity',
            'is_weekend','is_night','high_risk_location','high_velocity',
            'amount_x_velocity','risk_x_velocity']
print(f"✅ [PREPROCESS] {len(FEATURES)} features engineered")


# ══════════════════════════════════════════════════════════════
# STEP 3 — TRAIN / EVALUATE SPLIT
# ══════════════════════════════════════════════════════════════
X = df[FEATURES]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"✅ [SMOTE] Fraud recall before: {y_train.mean():.2%} → after: {y_train_bal.mean():.2%}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_bal)
X_test_s  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN
# ══════════════════════════════════════════════════════════════
print("\n🚀 [TRAIN] Training models...")
rf_model  = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
lr_model  = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
iso_model = IsolationForest(contamination=0.1, random_state=42)

rf_model.fit(X_train_s,  y_train_bal)
lr_model.fit(X_train_s,  y_train_bal)
iso_model.fit(X_train_s)
print("✅ [TRAIN] All models trained!")


# ══════════════════════════════════════════════════════════════
# STEP 5 — 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════
print("\n📊 [EVALUATE] 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rf = cross_val_score(rf_model, X_train_s, y_train_bal, cv=skf, scoring='roc_auc')
cv_lr = cross_val_score(lr_model, X_train_s, y_train_bal, cv=skf, scoring='roc_auc')
print(f"  Random Forest  CV AUC: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(f"  Logistic Reg   CV AUC: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")


# ══════════════════════════════════════════════════════════════
# STEP 6 — PREDICTIONS
# ══════════════════════════════════════════════════════════════
rf_pred  = rf_model.predict(X_test_s)
lr_pred  = lr_model.predict(X_test_s)
iso_raw  = iso_model.predict(X_test_s)
iso_pred = np.where(iso_raw == -1, 1, 0)

rf_proba  = rf_model.predict_proba(X_test_s)[:, 1]
lr_proba  = lr_model.predict_proba(X_test_s)[:, 1]
iso_score = -iso_model.decision_function(X_test_s)

rf_auc  = roc_auc_score(y_test, rf_proba)
lr_auc  = roc_auc_score(y_test, lr_proba)
iso_auc = roc_auc_score(y_test, iso_score)

print(f"\n  Random Forest  Test AUC : {rf_auc:.4f}")
print(f"  Logistic Reg   Test AUC : {lr_auc:.4f}")
print(f"  Isolation Forest AUC    : {iso_auc:.4f}")

print("\n" + "="*50 + "\nRANDOM FOREST REPORT\n" + "="*50)
print(classification_report(y_test, rf_pred))
print("="*50 + "\nLOGISTIC REGRESSION REPORT\n" + "="*50)
print(classification_report(y_test, lr_pred))


# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

# ── VIZ 1: EDA Dashboard ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('📊 Fraud Detection — EDA Dashboard', fontsize=16, color='#c9d1d9', fontweight='bold', y=1.01)

# 1a Fraud distribution
ax = axes[0, 0]
counts = df['is_fraud'].value_counts()
bars = ax.bar(['Legitimate', 'Fraud'], counts.values, color=[COLORS[0], COLORS[1]], edgecolor='#30363d', linewidth=1.2)
ax.set_title('Class Distribution', color='#c9d1d9')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,}', ha='center', color='#c9d1d9', fontsize=10)
ax.set_facecolor('#161b22')

# 1b Amount distribution
ax = axes[0, 1]
df[df['is_fraud']==0]['amount'].clip(upper=2000).hist(bins=50, ax=ax, alpha=0.7, color=COLORS[0], label='Legitimate')
df[df['is_fraud']==1]['amount'].clip(upper=2000).hist(bins=50, ax=ax, alpha=0.7, color=COLORS[1], label='Fraud')
ax.set_title('Transaction Amount Distribution', color='#c9d1d9')
ax.legend()
ax.set_xlabel('Amount ($)')
ax.set_facecolor('#161b22')

# 1c Transaction hour
ax = axes[0, 2]
df[df['is_fraud']==0]['time_hour'].hist(bins=24, ax=ax, alpha=0.7, color=COLORS[0], label='Legitimate')
df[df['is_fraud']==1]['time_hour'].hist(bins=24, ax=ax, alpha=0.7, color=COLORS[1], label='Fraud')
ax.set_title('Transaction Hour', color='#c9d1d9')
ax.legend()
ax.set_xlabel('Hour of Day')
ax.set_facecolor('#161b22')

# 1d Location risk
ax = axes[1, 0]
df[df['is_fraud']==0]['location_risk'].hist(bins=30, ax=ax, alpha=0.7, color=COLORS[0], label='Legitimate')
df[df['is_fraud']==1]['location_risk'].hist(bins=30, ax=ax, alpha=0.7, color=COLORS[1], label='Fraud')
ax.set_title('Location Risk Score', color='#c9d1d9')
ax.legend()
ax.set_facecolor('#161b22')

# 1e Velocity boxplot
ax = axes[1, 1]
data_box = [df[df['is_fraud']==0]['transaction_velocity'], df[df['is_fraud']==1]['transaction_velocity']]
bp = ax.boxplot(data_box, labels=['Legitimate','Fraud'], patch_artist=True)
for patch, color in zip(bp['boxes'], [COLORS[0], COLORS[1]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Transaction Velocity', color='#c9d1d9')
ax.set_facecolor('#161b22')

# 1f Merchant category
ax = axes[1, 2]
cat_fraud = df[df['is_fraud']==1]['merchant_category'].value_counts()
cat_fraud.plot(kind='bar', ax=ax, color=COLORS[1], alpha=0.8, edgecolor='#30363d')
ax.set_title('Fraud by Merchant Category', color='#c9d1d9')
ax.set_xlabel('')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_facecolor('#161b22')

plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 1: EDA Dashboard saved")


# ── VIZ 2: ROC Curves ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_score)

ax.plot(fpr_rf,  tpr_rf,  color=COLORS[0], lw=2, label=f'Random Forest (AUC={rf_auc:.3f})')
ax.plot(fpr_lr,  tpr_lr,  color=COLORS[2], lw=2, label=f'Logistic Regression (AUC={lr_auc:.3f})')
ax.plot(fpr_iso, tpr_iso, color=COLORS[3], lw=2, label=f'Isolation Forest (AUC={iso_auc:.3f})')
ax.plot([0,1],[0,1],'--', color='#8b949e', lw=1, label='Random Baseline')
ax.fill_between(fpr_rf, tpr_rf, alpha=0.08, color=COLORS[0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Fraud Detection Models', color='#c9d1d9', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 2: ROC Curves saved")


# ── VIZ 3: Precision-Recall Curves ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

pre_rf,  rec_rf,  _ = precision_recall_curve(y_test, rf_proba)
pre_lr,  rec_lr,  _ = precision_recall_curve(y_test, lr_proba)
pre_iso, rec_iso, _ = precision_recall_curve(y_test, iso_score)

ap_rf  = average_precision_score(y_test, rf_proba)
ap_lr  = average_precision_score(y_test, lr_proba)
ap_iso = average_precision_score(y_test, iso_score)

ax.plot(rec_rf,  pre_rf,  color=COLORS[0], lw=2, label=f'Random Forest (AP={ap_rf:.3f})')
ax.plot(rec_lr,  pre_lr,  color=COLORS[2], lw=2, label=f'Logistic Regression (AP={ap_lr:.3f})')
ax.plot(rec_iso, pre_iso, color=COLORS[3], lw=2, label=f'Isolation Forest (AP={ap_iso:.3f})')
ax.fill_between(rec_rf, pre_rf, alpha=0.08, color=COLORS[0])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Fraud Detection', color='#c9d1d9', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 3: Precision-Recall Curves saved")


# ── VIZ 4: Confusion Matrices ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Confusion Matrices', fontsize=14, color='#c9d1d9', fontweight='bold')

for ax, preds, title in zip(axes,
        [rf_pred, lr_pred, iso_pred],
        ['Random Forest', 'Logistic Regression', 'Isolation Forest']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'],
                linewidths=1, linecolor='#30363d')
    ax.set_title(title, color='#c9d1d9', fontweight='bold')
    ax.set_xlabel('Predicted', color='#c9d1d9')
    ax.set_ylabel('Actual', color='#c9d1d9')
    ax.set_facecolor('#161b22')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 4: Confusion Matrices saved")


# ── VIZ 5: Feature Importance ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

importance = rf_model.feature_importances_
sorted_idx = np.argsort(importance)
colors_bar = [COLORS[0] if v > np.median(importance) else COLORS[4] for v in importance[sorted_idx]]

ax.barh([FEATURES[i] for i in sorted_idx], importance[sorted_idx], color=colors_bar, edgecolor='#30363d')
ax.set_title('Feature Importance — Random Forest', color='#c9d1d9', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 5: Feature Importance saved")


# ── VIZ 6: 5-Fold CV Comparison ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

cv_data = [cv_rf, cv_lr]
labels  = ['Random Forest', 'Logistic Regression']
bp = ax.boxplot(cv_data, labels=labels, patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], [COLORS[0], COLORS[2]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers','caps','medians','fliers']:
    plt.setp(bp[element], color='#c9d1d9')

ax.set_title('5-Fold Cross-Validation AUC Scores', color='#c9d1d9', fontsize=13, fontweight='bold')
ax.set_ylabel('AUC Score')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('cross_validation.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 6: Cross-Validation saved")


# ── VIZ 7: Model Metrics Summary ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

models  = ['Random Forest', 'Logistic Reg', 'Isolation Forest']
metrics = {
    'AUC':       [rf_auc,  lr_auc,  iso_auc],
    'F1-Score':  [f1_score(y_test,rf_pred),   f1_score(y_test,lr_pred),   f1_score(y_test,iso_pred)],
    'Precision': [precision_score(y_test,rf_pred), precision_score(y_test,lr_pred), precision_score(y_test,iso_pred)],
    'Recall':    [recall_score(y_test,rf_pred),    recall_score(y_test,lr_pred),    recall_score(y_test,iso_pred)],
}

x     = np.arange(len(models))
width = 0.2
for i, (metric, vals) in enumerate(metrics.items()):
    ax.bar(x + i*width, vals, width, label=metric, color=COLORS[i], alpha=0.85, edgecolor='#30363d')

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.1)
ax.set_title('Model Performance Comparison', color='#c9d1d9', fontsize=13, fontweight='bold')
ax.set_ylabel('Score')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ VIZ 7: Model Comparison saved")


# ══════════════════════════════════════════════════════════════
# STEP 7 — REPORT (Real-time inference on new transactions)
# ══════════════════════════════════════════════════════════════
def report(new_transactions_raw):
    """Run inference on new transactions."""
    df_new = new_transactions_raw.copy()
    df_new['merchant_category_encoded'] = label_encoder.transform(df_new['merchant_category'])
    df_new['amount_log']         = np.log1p(df_new['amount'])
    df_new['is_night']           = ((df_new['time_hour'] >= 22) | (df_new['time_hour'] <= 5)).astype(int)
    df_new['high_risk_location'] = (df_new['location_risk'] > 0.5).astype(int)
    df_new['high_velocity']      = (df_new['transaction_velocity'] > 10).astype(int)
    df_new['amount_x_velocity']  = df_new['amount_log'] * df_new['transaction_velocity']
    df_new['risk_x_velocity']    = df_new['location_risk'] * df_new['transaction_velocity']
    X_new   = df_new[FEATURES]
    X_new_s = scaler.transform(X_new)
    preds   = rf_model.predict(X_new_s)
    probs   = rf_model.predict_proba(X_new_s)[:, 1]
    for i, row in df_new.iterrows():
        status = '🚨 FRAUD' if preds[i] == 1 else '✅ LEGIT'
        print(f"  Txn {i+1} | ${row['amount']:>8.2f} | {row['merchant_category']:<10} | "
              f"Risk={row['location_risk']:.2f} | {status} (conf={probs[i]:.2%})")

new_txns = pd.DataFrame({
    'amount':                    [50.0, 5000.0, 25.0, 1500.0],
    'time_hour':                 [14,   3,      10,   23],
    'merchant_category':         ['grocery','online','restaurant','atm'],
    'location_risk':             [0.1,  0.9,    0.2,  0.85],
    'user_age':                  [35,   28,     45,   22],
    'days_since_last_transaction': [5,  0,      12,   1],
    'transaction_velocity':      [2,    25,     1,    18],
    'is_weekend':                [0,    1,      0,    1],
})

print("\n🔍 [REPORT] Real-time Fraud Inference:")
report(new_txns)

print("\n✅ Full pipeline complete!")
print("   Outputs: eda_dashboard.png | roc_curves.png | precision_recall_curves.png")
print("            confusion_matrices.png | feature_importance.png")
print("            cross_validation.png  | model_comparison.png")
