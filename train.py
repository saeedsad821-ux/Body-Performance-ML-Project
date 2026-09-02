import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

print("Loading data...")
df = pd.read_csv("bodyPerformance.csv")

# Clean duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Define numeric columns (excluding categorical ones)
num_cols = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 
            'systolic', 'gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

# Winsorise outliers
for c in num_cols:
    Q1, Q3 = df[c].quantile([.25, .75])
    IQR = Q3 - Q1
    df[c] = df[c].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# Prepare features and target
df['gender'] = (df['gender'] == 'M').astype(int)
df['class_enc'] = df['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

FEATS = ['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%',
         'diastolic', 'systolic', 'gripForce', 'sit and bend forward_cm', 'sit-ups counts']

X = df[FEATS].values
y = df['class_enc'].values

print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training MLP Classifier (Neural Network)...")
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
model.fit(X_scaled, y)
print(f"Training accuracy: {model.score(X_scaled, y):.4f}")

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved as joblib files.")
