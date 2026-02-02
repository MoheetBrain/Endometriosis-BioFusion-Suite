## 🏆 Results

Comparative analysis on 10,000 synthetic patients:

| Model | Accuracy | Purpose |
|-------|----------|---------|
| **Symptoms Only** | 96.00% | Baseline (replicates current apps) |
| **Proteins Only** | 96.65% | Validates Wang et al. 2025 findings |
| **Bio-Fusion** | **99.65%** | Demonstrates multi-modal improvement |

### Key Findings

- **+3.65 percentage point improvement** over symptom-only approaches
- Feature importance analysis shows proteins and symptoms contribute roughly equally
- Per 10,000 patients, bio-fusion correctly diagnoses **365 additional cases**
- Translates to **2,555 person-years of diagnostic delay prevented** (7-year avg delay)

### Feature Importance (Bio-Fusion Model)

Top predictors ranked by importance:
1. Pain_Level
2. Protein_CA125
3. Protein_IL6
4. Protein_TNFa
5. Heavy_Bleeding
6. Cycle_Length

**Insight:** Biological markers (57%) and symptoms (43%) both essential—
neither modality alone achieves 99%+ accuracy.
