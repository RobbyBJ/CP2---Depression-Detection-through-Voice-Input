import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier

# ================= CONFIGURATION =================
TRAIN_CSV = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
MODEL_DIR = r"C:\Users\User\Desktop\CP2\tuned_models"
OUTPUT_PATH = r"C:\Users\User\Desktop\CP2\ensemble_models\voting_ensemble.pkl"

# Weights 
# KNN (Acc) gets 2.0
# LogReg (Sens) gets 1.2
# SVM (Balance) gets 1.0
# RF (Backup) gets 0.8
VOTING_WEIGHTS = {'KNN': 2.0, 'LogisticRegression': 1.2, 'SVM': 1.0, 'RandomForest': 0.8}
# =================================================

def build_production_model():
    print("🚀 BUILDING SINGLE APP MODEL...")

    # 1. Load Training Data
    if not os.path.exists(TRAIN_CSV):
        print("❌ Dataset not found.")
        return
    
    df_train = pd.read_csv(TRAIN_CSV)
    X_train = df_train.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df_train['PHQ8_Binary']
    
    # Safety Clean
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Load the Tuned "Specialists"
    estimators = []
    weights_list = []
    
    print("   LOADING SPECIALISTS:")
    for name, weight in VOTING_WEIGHTS.items():
        path = os.path.join(MODEL_DIR, f"{name}_tuned_v2.pkl")
        if os.path.exists(path):
            print(f"   ✅ Loading {name} (Weight: {weight})")
            model = joblib.load(path)
            estimators.append((name, model))
            weights_list.append(weight)
        else:
            print(f"   ⚠️ Warning: {name} not found. Skipping.")

    if not estimators:
        print("❌ No models found!")
        return

    # 3. Create the Super Model (VotingClassifier)
    # This bundles them into one object.
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',         # Soft voting averages probabilities (Better than Hard voting)
        weights=weights_list,
        n_jobs=-1
    )

    # 4. Fit on FULL Data
    print("\n   🧠 Training final ensemble on full dataset...")
    voting_clf.fit(X_train, y_train)

    # 5. Save the Single File
    joblib.dump(voting_clf, OUTPUT_PATH)
    print(f"\n🎉 SUCCESS! Model saved to:\n   {OUTPUT_PATH}")
    print("   (This is the ONLY file you need for your app)")

if __name__ == "__main__":
    build_production_model()