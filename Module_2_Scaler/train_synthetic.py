import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- PART 1: SCIENTIFIC DATA GENERATION (The "Wang et al." Simulation) ---
print("🧪 Simulating Clinical Cohort (n=10,000)...")

np.random.seed(42)
n_per_group = 5000

# GROUP A: ENDOMETRIOSIS PATIENTS (The "Sick" Profile)
# High Inflammation (IL-6 > 30), High Pain, Irregular Cycles
endo_data = {
    'Protein_IL6': np.random.normal(35, 12, n_per_group),    # Elevated
    'Protein_CA125': np.random.normal(75, 30, n_per_group),  # Highly Elevated
    'Protein_TNFa': np.random.normal(22, 8, n_per_group),    # Elevated
    'Pain_Level': np.random.normal(8, 1.5, n_per_group),     # High Pain (Mean 8)
    'Cycle_Length': np.random.normal(25, 6, n_per_group),    # Irregular
    'Heavy_Bleeding': np.random.choice([0, 1], n_per_group, p=[0.2, 0.8]), # 80% have heavy bleeding
    'Diagnosis': 1
}

# GROUP B: HEALTHY CONTROLS (The "Healthy" Profile)
# Low Inflammation, Low Pain, Regular Cycles
healthy_data = {
    'Protein_IL6': np.random.normal(12, 8, n_per_group),     # Normal
    'Protein_CA125': np.random.normal(22, 15, n_per_group),  # Normal
    'Protein_TNFa': np.random.normal(12, 6, n_per_group),    # Normal
    'Pain_Level': np.random.normal(3, 2, n_per_group),       # Low/Normal Pain
    'Cycle_Length': np.random.normal(28, 2, n_per_group),    # Regular
    'Heavy_Bleeding': np.random.choice([0, 1], n_per_group, p=[0.8, 0.2]), # Only 20% heavy bleeding
    'Diagnosis': 0
}

# Merge & Shuffle
df_endo = pd.DataFrame(endo_data)
df_healthy = pd.DataFrame(healthy_data)
df = pd.concat([df_endo, df_healthy]).sample(frac=1, random_state=42).reset_index(drop=True)

# Clip values to realistic ranges (No negative pain)
df['Pain_Level'] = df['Pain_Level'].clip(0, 10)
df['Protein_IL6'] = df['Protein_IL6'].clip(0, 100)

print(f"✅ Generated {len(df)} patients (50% Endo / 50% Healthy)")
df.to_csv('Module_2_Scaler/scientific_endo_data.csv', index=False)


# --- PART 2: THE TRIPLE THREAT TEST (Comparison) ---
print("\n🧠 Training 3 Comparative Models...")

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DEFINE THE FEATURE SETS
features_symptoms = ['Pain_Level', 'Cycle_Length', 'Heavy_Bleeding']
features_proteins = ['Protein_IL6', 'Protein_CA125', 'Protein_TNFa']
features_fusion   = features_symptoms + features_proteins

# TRAIN MODEL 1: SYMPTOMS ONLY (The "Old Way")
clf_sym = RandomForestClassifier(n_estimators=100, random_state=42)
clf_sym.fit(X_train[features_symptoms], y_train)
acc_sym = accuracy_score(y_test, clf_sym.predict(X_test[features_symptoms]))

# TRAIN MODEL 2: PROTEINS ONLY (The "Lab Test")
clf_bio = RandomForestClassifier(n_estimators=100, random_state=42)
clf_bio.fit(X_train[features_proteins], y_train)
acc_bio = accuracy_score(y_test, clf_bio.predict(X_test[features_proteins]))

# TRAIN MODEL 3: BIO-FUSION (The "Moheet Solution")
clf_fusion = RandomForestClassifier(n_estimators=100, random_state=42)
clf_fusion.fit(X_train[features_fusion], y_train)
acc_fusion = accuracy_score(y_test, clf_fusion.predict(X_test[features_fusion]))

# --- PART 3: THE VERDICT ---
print(f"\n📊 RESULTS TABLE:")
print(f"1. Symptoms Only (Baseline):  {acc_sym*100:.2f}%")
print(f"2. Proteins Only (Wang 2025): {acc_bio*100:.2f}%")
print(f"3. Bio-Fusion (Your Model):   {acc_fusion*100:.2f}%")

lift = (acc_fusion - acc_sym) * 100
print(f"\n🚀 IMPROVEMENT: Adding biomarkers improved diagnosis by +{lift:.1f} points!")

# --- VISUALIZATION ---
models = ['Symptoms Only', 'Proteins Only', 'Bio-Fusion']
accuracies = [acc_sym, acc_bio, acc_fusion]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['gray', 'blue', 'crimson'])
plt.ylim(0.8, 1.0) # Zoom in to show difference
plt.title(f"Diagnostic Accuracy Comparison\nBio-Fusion adds +{lift:.1f}% Accuracy over Symptoms", fontsize=14)
plt.ylabel("Accuracy Score")

# Add numbers on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval*100:.1f}%", va='bottom', ha='center', fontsize=12, fontweight='bold')

plt.savefig('Module_2_Scaler/model_comparison.png')
print("✅ Chart saved: 'Module_2_Scaler/model_comparison.png'")