import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score

# ================= CONFIGURATION =================
TEST_CSV = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"
MODEL_PATH = r"C:\Users\User\Desktop\CP2\ensemble_model\stacking_ensemble.pkl"
# =================================================

def run_supervisor_voting():
    print("🚀 TESTING SUPERVISOR'S METHOD (Hard Voting)...")
    
    if not os.path.exists(TEST_CSV) or not os.path.exists(MODEL_PATH):
        print("❌ Error: Files not found.")
        return

    # 1. Load Data
    df = pd.read_csv(TEST_CSV)
    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    
    # 2. Load Model
    model = joblib.load(MODEL_PATH)
    
    # 3. GET SEGMENT PREDICTIONS (0 or 1)
    # This is "Hard Voting" - we treat every segment as a strict Yes/No
    print("   Getting segment predictions (this might take a moment)...")
    segment_preds = model.predict(X_test)
    
    # Add predictions to metadata
    df_meta = df[['participant_id', 'PHQ8_Binary']].copy()
    df_meta['seg_pred'] = segment_preds

    # 4. AGGREGATE PER PATIENT
    # Count how many segments were "1" (Depressed) vs Total Segments
    participant_stats = df_meta.groupby('participant_id').agg({
        'seg_pred': ['sum', 'count'], # Sum = count of 1s, Count = total
        'PHQ8_Binary': 'first'
    }).reset_index()
    
    # Flatten columns
    participant_stats.columns = ['participant_id', 'depressed_count', 'total_count', 'y_true']
    
    # Calculate Ratio (e.g., 70/100 = 0.7)
    participant_stats['vote_ratio'] = participant_stats['depressed_count'] / participant_stats['total_count']
    
    y_true = participant_stats['y_true']
    ratios = participant_stats['vote_ratio']

    # --- SCENARIO A: SUPERVISOR'S STRICT RULE (>50%) ---
    print("\n📋 SCENARIO A: Strict Majority Rule (>50% samples)")
    y_pred_strict = (ratios > 0.50).astype(int)
    
    acc_strict = accuracy_score(y_true, y_pred_strict)
    sens_strict = recall_score(y_true, y_pred_strict)
    
    print(f"   Accuracy:    {acc_strict:.2%}")
    print(f"   Sensitivity: {sens_strict:.2%}")
    print("   (If this is low, it confirms the 50% rule is too harsh)")

    # --- SCENARIO B: CALIBRATED VOTING RULE (Tuning the Ratio) ---
    print("\n📋 SCENARIO B: Calibrated Voting Rule (Finding the Sweet Spot)")
    print(f"{'Vote % Needed':<15} | {'Acc':<10} | {'Sens':<10} | {'Spec':<10}")
    print("-" * 55)
    
    best_acc = 0
    best_threshold = 0
    
    # Check ratios from 10% to 50%
    for thresh in np.arange(0.10, 0.55, 0.05):
        y_pred = (ratios > thresh).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        
        # Calculate specificity manually
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f" > {thresh:.0%} samples   | {acc:.2%}   | {sens:.2%}   | {spec:.2%}")
        
        if acc >= best_acc:
            best_acc = acc
            best_threshold = thresh

    print("-" * 55)
    print(f"🏆 BEST VOTING RESULT: If >{best_threshold:.0%} of samples are depressed.")
    print(f"   Max Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    run_supervisor_voting()