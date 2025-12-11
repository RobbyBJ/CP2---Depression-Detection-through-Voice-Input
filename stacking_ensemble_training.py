import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ================= CONFIGURATION =================
# 1. Path to your Training Data (Same as before)
TRAIN_CSV = r"C:\Users\User\Desktop\depression_train_opensmile.csv"

# 2. Path where you saved the TUNED models
TUNED_MODEL_DIR = r"C:\Users\User\Desktop\CP2\tuned_models_opensmile"

# 3. Output for the Final Ensemble
OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\final_ensemble_opensmile"
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tuned_models():
    print("search for tuned models...")
    estimators = []
    
    # We look for the .pkl files created by the Tuning Script
    # Expected names: SVM_tuned_v8.pkl, XGBoost_tuned_v8.pkl, etc.
    if not os.path.exists(TUNED_MODEL_DIR):
        print("❌ Tuned model directory not found.")
        return []

    for f in os.listdir(TUNED_MODEL_DIR):
        if f.endswith("tuned_v8.pkl"):
            name = f.replace("_tuned_v8.pkl", "")
            path = os.path.join(TUNED_MODEL_DIR, f)
            
            try:
                model = joblib.load(path)
                estimators.append((name, model))
                print(f"   ✅ Loaded: {name}")
            except:
                print(f"   ❌ Failed to load: {f}")
                
    return estimators

def train_stacking():
    print("🚀 STARTING STACKING ENSEMBLE TRAINING...")
    
    # 1. Load Data
    df = pd.read_csv(TRAIN_CSV)
    X_train = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df['PHQ8_Binary']
    
    # 2. Load Base Learners
    estimators = load_tuned_models()
    if not estimators:
        print("❌ No models found! Run 'train_tuned_opensmile.py' first.")
        return

    # 3. Define Stacking Classifier
    # Final Estimator = Logistic Regression (Standard for Stacking)
    # cv=5 ensures we don't overfit while training the meta-learner
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=False # False = Meta-learner only sees predictions, not original features
    )
    
    print("\n⏳ Training Meta-Learner (This may take a while)...")
    try:
        stacking_clf.fit(X_train, y_train)
        
        # 4. Save
        save_path = os.path.join(OUTPUT_DIR, "Final_Stacking_Ensemble_v8.pkl")
        joblib.dump(stacking_clf, save_path)
        print(f"\n✅ ENSEMBLE TRAINED & SAVED: {save_path}")
        print("👉 Now test this model using 'test_v8_voting_final.py'!")
        
    except Exception as e:
        print(f"❌ Stacking Failed: {e}")

if __name__ == "__main__":
    train_stacking()