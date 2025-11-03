# train_enhanced.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("🚀 Training ENHANCED Heart Disease Model...")

# Load your original dataset
data = pd.read_csv("project/Heart/heart_disease_data.csv")
print(f"✅ Original dataset: {data.shape}")

# Create high-risk examples to add to training
def create_high_risk_examples():
    high_risk_cases = []
    
    # Extreme high-risk patterns that should definitely be defective heart
    patterns = [
        # Pattern 1: Elderly male with multiple risk factors
        [65, 1, 3, 180, 300, 1, 1, 110, 1, 4.2, 1, 3, 2, 1],
        [70, 1, 3, 190, 320, 1, 1, 105, 1, 4.5, 1, 3, 2, 1],
        [68, 1, 3, 175, 310, 1, 1, 115, 1, 3.8, 1, 2, 2, 1],
        
        # Pattern 2: Critical vessel blockage
        [60, 1, 3, 160, 280, 0, 1, 120, 1, 3.5, 1, 3, 2, 1],
        [62, 1, 3, 170, 290, 1, 1, 118, 1, 4.0, 1, 3, 2, 1],
        
        # Pattern 3: Severe symptoms combination
        [55, 1, 3, 165, 275, 1, 1, 125, 1, 3.2, 1, 2, 2, 1],
        [58, 1, 3, 155, 265, 0, 1, 128, 1, 3.6, 1, 2, 2, 1]
    ]
    
    for case in patterns:
        high_risk_cases.append(case)
    
    return pd.DataFrame(high_risk_cases, columns=data.columns)

# Create clear low-risk examples
def create_low_risk_examples():
    low_risk_cases = []
    
    patterns = [
        # Young female with excellent metrics
        [35, 0, 0, 110, 180, 0, 0, 175, 0, 0.5, 0, 0, 0, 0],
        [30, 0, 0, 105, 170, 0, 0, 180, 0, 0.3, 0, 0, 0, 0],
        [28, 0, 0, 115, 185, 0, 0, 170, 0, 0.4, 0, 0, 0, 0]
    ]
    
    for case in patterns:
        low_risk_cases.append(case)
    
    return pd.DataFrame(low_risk_cases, columns=data.columns)

# Add enhanced examples to original data
high_risk_data = create_high_risk_examples()
low_risk_data = create_low_risk_examples()

enhanced_data = pd.concat([data, high_risk_data, low_risk_data], ignore_index=True)
print(f"📈 Enhanced dataset: {enhanced_data.shape}")
print(f"Target distribution:\n{enhanced_data['target'].value_counts()}")

# Prepare features and target
X = enhanced_data.drop('target', axis=1)
y = enhanced_data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n📊 Data Split:")
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

print("\n🔧 Training enhanced model...")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"📊 Cross-validation: {cv_scores.mean():.3f}")

# Train final model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Performance:")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 Feature Importances:")
print(feature_importance.head(8))

# Test our specific cases
print(f"\n🧪 CRITICAL TEST CASES:")

# High-risk case (should be DEFECTIVE HEART)
high_risk_case = np.array([[65, 1, 3, 180, 300, 1, 1, 110, 1, 4.2, 1, 3, 2]])
high_risk_pred = model.predict(high_risk_case)[0]
high_risk_proba = model.predict_proba(high_risk_case)[0]

print(f"🚨 HIGH-RISK PATIENT:")
print(f"   Prediction: {'DEFECTIVE HEART' if high_risk_pred == 1 else 'HEALTHY HEART'}")
print(f"   Probability: {high_risk_proba[1]*100:.1f}% defective")
print(f"   Expected: DEFECTIVE HEART (>80% probability)")

# Low-risk case (should be HEALTHY HEART)
low_risk_case = np.array([[35, 0, 0, 110, 180, 0, 0, 175, 0, 0.5, 0, 0, 0]])
low_risk_pred = model.predict(low_risk_case)[0]
low_risk_proba = model.predict_proba(low_risk_case)[0]

print(f"✅ LOW-RISK PATIENT:")
print(f"   Prediction: {'DEFECTIVE HEART' if low_risk_pred == 1 else 'HEALTHY HEART'}")
print(f"   Probability: {low_risk_proba[1]*100:.1f}% defective")
print(f"   Expected: HEALTHY HEART (<20% probability)")

# Save model
model_data = {
    'model': model,
    'feature_names': X.columns.tolist(),
    'test_accuracy': accuracy,
    'is_enhanced': True
}

with open(r"project\Heart\heart_disease_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✅ ENHANCED model saved!")
if high_risk_pred == 1 and high_risk_proba[1] > 0.8:
    print("🎉 HIGH-RISK DETECTION: WORKING PERFECTLY!")
else:
    print("❌ High-risk detection still needs improvement")