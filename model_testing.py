import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, recall_score, precision_score
)

# ================= CONFIGURATION =================
# Make sure this points to your V8 Segment-Level Test Set
TEST_DATASET = r"C:\Users\User\Desktop\depression_test_opensmile.csv"
MODEL_DIR = r"C:\Users\User\Desktop\CP2\final_ensemble_advanced"
OUTPUT_RESULTS = r"C:\Users\User\Desktop\ensemble_results_advanced.csv"
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def align_features(model, X):
    """Aligns test features to training features."""
    if hasattr(model, "feature_names_in_"):
        trained_features = model.feature_names_in_
        # Reindex ensures columns match exact training order/names
        X_aligned = X.reindex(columns=trained_features, fill_value=0)
        return X_aligned
    return X

def evaluate_participant_level(model, X_test, df_meta):
    """
    Performs Majority Voting:
    1. Predicts every segment.
    2. Groups by Patient ID.
    3. If >50% of segments are 'Depressed', the Patient is 'Depressed'.
    """
    # 1. Segment Predictions
    seg_preds = model.predict(X_test)
    
    # 2. Combine with Metadata
    results = df_meta.copy()
    results['seg_pred'] = seg_preds
    
    # 3. Vote (Calculate % of segments predicted as Class 1)
    participant_scores = results.groupby('participant_id')['seg_pred'].mean()
    
    # Threshold 0.5 = Majority Vote
    participant_preds = (participant_scores > 0.5).astype(int)
    
    # Get True Labels (Take first label for each ID)
    true_labels = results.groupby('participant_id')['PHQ8_Binary'].first()
    
    # 4. Calculate Clinical Metrics
    acc = accuracy_score(true_labels, participant_preds)
    f1 = f1_score(true_labels, participant_preds)
    sens = recall_score(true_labels, participant_preds)
    prec = precision_score(true_labels, participant_preds, zero_division=0)
    spec = calculate_specificity(true_labels, participant_preds)
    
    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1-Score": f1,
        "Precision": prec
    }

def main():
    print("🚀 STARTING FINAL VOTING EVALUATION (V8)...")
    print(f"📘 Data: {TEST_DATASET}")
    
    if not os.path.exists(TEST_DATASET):
        print("❌ Dataset not found.")
        return

    # Load Data
    df = pd.read_csv(TEST_DATASET)
    
    # Separate Features vs Metadata
    # We need participant_id for Voting!
    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    df_meta = df[['participant_id', 'PHQ8_Binary']]
    
    print(f"✅ Loaded {len(X_test)} segments from {df_meta['participant_id'].nunique()} participants.")

    # Find Models
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model dir not found: {MODEL_DIR}")
        return
        
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not model_files:
        print("⚠️ No models found.")
        return

    results_log = []

    print("\n⚔️ EVALUATING PARTICIPANT-LEVEL DIAGNOSIS...")
    print(f"{'Model':<25} | {'Acc':<8} | {'Sens':<8} | {'Spec':<8} | {'F1':<8}")
    print("-" * 70)

    for f in model_files:
        model_path = os.path.join(MODEL_DIR, f)
        model_name = f.replace(".pkl", "")
        
        try:
            model = joblib.load(model_path)
            
            # Align Features
            X_test_aligned = align_features(model, X_test)
            
            # Run Voting Logic
            metrics = evaluate_participant_level(model, X_test_aligned, df_meta)
            
            print(f"{model_name:<25} | {metrics['Accuracy']:.2%} | {metrics['Sensitivity']:.2%} | {metrics['Specificity']:.2%} | {metrics['F1-Score']:.2f}")
            
            entry = {"Model": model_name}
            entry.update(metrics)
            results_log.append(entry)
            
        except Exception as e:
            print(f"❌ Error {model_name}: {e}")

    # Save
    if results_log:
        res_df = pd.DataFrame(results_log).sort_values(by="Sensitivity", ascending=False)
        res_df.to_csv(OUTPUT_RESULTS, index=False)
        print(f"\n✅ Final Voting Results saved to: {OUTPUT_RESULTS}")
        print("\n🏆 LEADERBOARD (Sorted by Sensitivity):")
        print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()