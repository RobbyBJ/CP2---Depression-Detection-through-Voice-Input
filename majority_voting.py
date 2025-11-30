import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ================= CONFIGURATION =================
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# THRESHOLD: The % of "Depressed" segments needed to diagnose a patient.
# 0.5 = Majority Vote (Standard)
# 0.3 = Conservative (If 30% of clips sound depressed, flag them) -> Increases Sensitivity
VOTING_THRESHOLD = 0.3
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def run_majority_voting():
    print("🚀 LOADING DATASET FOR PATIENT DIAGNOSIS...")
    df = pd.read_csv(INPUT_CSV)
    
    # --- 1. GROUP-AWARE SPLIT (Recreating your Test Set) ---
    participant_data = df[['participant_id', 'PHQ8_Binary']].drop_duplicates()
    
    train_ids, test_ids = train_test_split(
        participant_data['participant_id'], 
        test_size=TEST_SIZE, 
        stratify=participant_data['PHQ8_Binary'],
        random_state=RANDOM_STATE
    )
    
    train_df = df[df['participant_id'].isin(train_ids)]
    test_df = df[df['participant_id'].isin(test_ids)]
    
    X_train = train_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_train = train_df['PHQ8_Binary']
    
    # KEEP participant_id in Test Set so we can group later
    X_test = test_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_test_segments = test_df['PHQ8_Binary']
    test_participant_ids = test_df['participant_id'].values # Store IDs for voting
    
    print(f"   Training on {len(X_train)} segments...")
    print(f"   Testing on {len(test_ids)} participants ({len(X_test)} segments)")

    # --- 2. TRAIN BEST MODEL (Logistic Regression) ---
    # We use the best params you found: C=0.1, solver=liblinear, class_weight=balanced
    print("\n⚔️ TRAINING LOGISTIC REGRESSION...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=0.1, 
            solver='liblinear', 
            class_weight='balanced', 
            random_state=RANDOM_STATE
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # --- 3. PREDICT SEGMENTS ---
    print("🔮 PREDICTING SEGMENTS...")
    # Get probability of being "Depressed" (Class 1)
    # This is better than hard predictions for voting
    y_pred_probs = pipeline.predict_proba(X_test)[:, 1] 
    
    # Create a DataFrame for Voting
    results_df = pd.DataFrame({
        'participant_id': test_participant_ids,
        'true_label': y_test_segments.values,
        'pred_prob': y_pred_probs,
        'pred_label_segment': (y_pred_probs > 0.5).astype(int) # Standard 0.5 cutoff for segments
    })
    
    # --- 4. MAJORITY VOTING (The Magic Step) ---
    print(f"\n🗳️ PERFORMING MAJORITY VOTING (Threshold = {VOTING_THRESHOLD})...")
    
    patient_results = []
    
    # Group by Patient
    for pid, group in results_df.groupby('participant_id'):
        # Percentage of segments predicted as "Depressed"
        depressed_ratio = group['pred_label_segment'].mean()
        
        # Does this patient cross the threshold?
        final_prediction = 1 if depressed_ratio >= VOTING_THRESHOLD else 0
        true_label = group['true_label'].iloc[0] # All segments have same true label
        
        patient_results.append({
            'participant_id': pid,
            'Depressed_Segment_Ratio': round(depressed_ratio, 3),
            'Final_Prediction': final_prediction,
            'True_Label': true_label
        })
        
    patient_df = pd.DataFrame(patient_results)
    
    # --- 5. FINAL PATIENT-LEVEL METRICS ---
    y_true_final = patient_df['True_Label']
    y_pred_final = patient_df['Final_Prediction']
    
    acc = accuracy_score(y_true_final, y_pred_final)
    f1 = f1_score(y_true_final, y_pred_final)
    rec = classification_report(y_true_final, y_pred_final, output_dict=True)['1']['recall']
    spec = calculate_specificity(y_true_final, y_pred_final)
    
    print("\n" + "="*50)
    print(f"🏥 FINAL PATIENT DIAGNOSIS RESULTS (N={len(patient_df)})")
    print("="*50)
    print(f"ACCURACY:    {acc:.4f}")
    print(f"F1-SCORE:    {f1:.4f}")
    print(f"SENSITIVITY: {rec:.4f} (Recall)")
    print(f"SPECIFICITY: {spec:.4f}")
    print("-" * 50)
    print("\nDetailed Patient Predictions:")
    print(patient_df.sort_values(by='Depressed_Segment_Ratio', ascending=False).head(10))
    
    patient_df.to_csv("final_patient_diagnosis.csv", index=False)
    print("\n✅ Saved patient-level report to 'final_patient_diagnosis.csv'")

if __name__ == "__main__":
    run_majority_voting()