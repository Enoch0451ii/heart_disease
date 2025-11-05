# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("🚀 Training Enhanced Heart Disease Model...")

# Load your dataset
df = pd.read_csv('project\Heart\heart_disease_data.csv')
print(f"Dataset shape: {df.shape}")

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model with better parameters for higher confidence
model = RandomForestClassifier(
    n_estimators=200,  # More trees for better performance
    max_depth=15,       # Deeper trees
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

# Test the model with EXTREME cases
print("\n🧪 Testing model with EXTREME cases:")

# EXTREME High-risk case (should predict 1 - heart disease with high probability)
extreme_high_risk = [70, 1, 3, 200, 400, 1, 2, 100, 1, 5.0, 2, 3, 3]  # More extreme values
high_prob = model.predict_proba([extreme_high_risk])[0]
high_pred = model.predict([extreme_high_risk])[0]
print(f"EXTREME High-risk case:")
print(f"  Prediction: {high_pred}")
print(f"  Probabilities: {high_prob}")
print(f"  Heart Disease Probability: {high_prob[1]:.3f}")

# EXTREME Low-risk case (should predict 0 - no heart disease with low probability)  
extreme_low_risk = [30, 0, 0, 100, 150, 0, 0, 190, 0, 0.1, 0, 0, 0]  # More extreme values
low_prob = model.predict_proba([extreme_low_risk])[0]
low_pred = model.predict([extreme_low_risk])[0]
print(f"\nEXTREME Low-risk case:")
print(f"  Prediction: {low_pred}")
print(f"  Probabilities: {low_prob}")
print(f"  Heart Disease Probability: {low_prob[1]:.3f}")

# Test the previous cases too
print(f"\n🧪 Testing previous cases:")
previous_high_risk = [65, 1, 3, 180, 300, 1, 1, 110, 1, 4.2, 1, 3, 2]
prev_high_prob = model.predict_proba([previous_high_risk])[0]
prev_high_pred = model.predict([previous_high_risk])[0]
print(f"Previous High-risk: Prediction {prev_high_pred}, Prob {prev_high_prob[1]:.3f}")

previous_low_risk = [35, 0, 0, 110, 180, 0, 0, 175, 0, 0.5, 0, 0, 1]
prev_low_prob = model.predict_proba([previous_low_risk])[0]
prev_low_pred = model.predict([previous_low_risk])[0]
print(f"Previous Low-risk: Prediction {prev_low_pred}, Prob {prev_low_prob[1]:.3f}")

# Determine which probability index is correct
if high_pred == 1 and high_prob[1] > high_prob[0]:
    print("✅ probability[1] = Heart Disease probability")
    prob_index = 1
else:
    print("🔄 probability[0] = Heart Disease probability")  
    prob_index = 0

# Save model with correct probability index
model_data = {
    'model': model,
    'feature_names': list(X.columns),
    'probability_index': prob_index,
    'test_accuracy': model.score(X_test, y_test),
    'extreme_high_risk_case': extreme_high_risk,
    'extreme_low_risk_case': extreme_low_risk
}

joblib.dump(model_data, 'heart_model.joblib')
print(f"\n✅ Model saved with accuracy: {model_data['test_accuracy']:.3f}")
print(f"✅ Using probability index: {prob_index} for heart disease")
print(f"✅ Extreme high-risk probability: {high_prob[1]:.3f}")
print(f"✅ Extreme low-risk probability: {low_prob[1]:.3f}")