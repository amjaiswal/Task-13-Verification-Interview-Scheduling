# Task 13 — Verification & Interview Scheduling
## Proctoring False-Positive (FP) Reduction — PlaceMux AI/ML Engineer

### Objective
Reduce false positives in the proctoring "suspicious behaviour" detector vs a
simple rule-based baseline, with real precision / recall / FP-rate numbers on
held-out data — not just claims.

### Files
- `Task13_FP_Reduction.ipynb` — full pipeline (data load → clean → baseline →
  model → tuning → evaluation → explainability → robustness → live demo).
- `Students_suspicious_behaviors_detection_dataset_V1.csv` — sample dataset.

### Pipeline summary
1. **Missing values**: `head_pose` (569 missing) and `gaze_direction` (1051
   missing) filled as `"unknown"` — a face/gaze not being detected is itself a
   signal, not something to drop.
2. **Baseline**: rule-based flag (`phone_present OR hand_obj_interaction OR
   gaze off script`).
3. **Model**: Random Forest, threshold tuned on validation set to minimise
   false positives while keeping recall ≥ 0.80.
4. **Results (test set)**:
   - Baseline FPR: 0.130 → Model FPR: 0.000
   - Model Recall: ~0.90, AUC: 1.0 on this sample data
5. **Explainability**: feature importances + one-example plain-English
   walkthrough for every flagged case.
6. **Known limitation**: this sample dataset's labels are largely rule-derived
   from a handful of dominant features (see the leakage-check cell), so
   clean-data metrics are a ceiling, not a real-world guarantee. A 5%-noise
   robustness check is included to show behaviour closer to real, messier
   sensor data.
7. **Live inference**: `predict_one(row_dict)` — takes one session's signals,
   returns a prediction, probability, and plain-English reason. Handles
   missing fields, unseen categories, and no-face-detected frames without
   crashing (falls back to "flagged for manual review").

### Dependency
Upstream: flagged-session data from the proctoring capture layer (face, hand,
gaze, phone signals). If that feed is late or partially missing, `predict_one`
degrades gracefully instead of failing.

### How to run
```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook Task13_FP_Reduction.ipynb
```
