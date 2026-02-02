import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- PART 1: GENESIS (Bio-Fusion Data Generation) ---
print("🧪 Generative AI: Creating 10,000 Bio-Fusion Patient Records...")

np.random.seed(42)
n_samples = 10000

# 1. BIOLOGICAL DATA (The "Wang et al." replication)
# Simulating inflammatory proteins found in menstrual blood
# These are the "Hidden Signals" current apps miss.
bio_markers = {
    'Protein_IL6': np.random.normal(30, 10, n_samples),   # Inflammatory Marker
    'Protein_CA125': np.random.normal(25, 15, n_samples), # Standard Endometriosis Marker
    'Protein_TNFa': np.random.normal(15, 5, n_samples)    # Tumor Necrosis Factor
}

# 2. SYMPTOM DATA (The "Lucy App" replication)
symptoms = {
    'Pain_Level': np.random.randint(0, 11, n_samples),
    'Cycle_Length': np.random.normal(28, 5, n_samples).astype(int),
    'Heavy_Bleeding': np.random.randint(0, 2, n_samples)  # 0=No, 1=Yes
}

# Combine into one dataset
data = {**bio_markers, **symptoms}
df = pd.DataFrame(data)

# --- THE LOGIC OF DISEASE ---
# Logic: Endometriosis is High Inflammation (Proteins) + High Pain (Symptoms)
# This simulates the "Wang et al." finding that proteins improve diagnosis.
risk_score = (
    (df['Protein_IL6'] * 1.5) + 
    (df['Protein_CA125'] * 0.5) + 
    (df['Pain_Level'] * 2) + 
    (df['Heavy_Bleeding'] * 10) +
    np.random.normal(0, 10, n_samples) # Biological noise
)

# Threshold for diagnosis
df['Diagnosis'] = (risk_score > 85).astype(int)

# Save the Asset
df.to_csv('Module_2_Scaler/bio_fusion_endo_data.csv', index=False)
print("✅ Data Created: 'bio_fusion_endo_data.csv' (Proteins + Symptoms)")

# --- PART 2: THE BRAIN ---
print("\n🧠 Training Bio-Fusion Model...")

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# --- PART 3: THE VERDICT ---
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("------------------------------------------------")
print(f"🚀 BIO-FUSION ACCURACY: {acc*100:.2f}%")
print("------------------------------------------------")

# Visualize: Prove that Proteins Matter
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Predictors (Notice how Proteins compete with Pain):")
print(importances.sort_values(ascending=False).head(5))

# Save the Chart
plt.figure(figsize=(10,5))
importances.sort_values().plot(kind='barh', color='crimson')
plt.title("Bio-Fusion: Do Proteins predict better than Pain?")
plt.xlabel("Importance in Diagnosis")
plt.tight_layout()
plt.savefig('Module_2_Scaler/feature_importance.png')
print("📊 Chart saved as 'feature_importance.png'")