import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# ================= CONFIGURATION =================
TEST_CSV = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"
MODEL_PATH = r"C:\Users\User\Desktop\CP2\ensemble_models\stacking_ensemble.pkl"

OPTIMAL_THRESHOLD = 0.40

# Where to save the images
OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\report_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

def plot_cm(y_true, y_pred, title, filename):
    """Helper function to draw and save a Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for clearer visualization
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    
    # Draw Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Healthy', 'Predicted Depressed'],
                yticklabels=['Actual Healthy', 'Actual Depressed'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Matrix: {save_path}")
    plt.close()

def main():
    print("🚀 GENERATING CONFUSION MATRICES...")

    # 1. Load Data
    if not os.path.exists(TEST_CSV) or not os.path.exists(MODEL_PATH):
        print("❌ Error: Missing Test CSV or Model file.")
        return

    df = pd.read_csv(TEST_CSV)
    
    print("   Loading Model & Data...")
    model = joblib.load(MODEL_PATH)
    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    
    # Get Segment Probabilities
    segment_probs = model.predict_proba(X_test)[:, 1]
    
    # Add to dataframe to aggregate by Participant
    df['prob'] = segment_probs
    
    # Group by Participant (The crucial step for valid evaluation)
    participant_level = df.groupby('participant_id').agg({
        'prob': 'mean',          # Average probability of all segments
        'PHQ8_Binary': 'first'   # The actual label
    }).reset_index()

    y_true = participant_level['PHQ8_Binary']
    y_probs = participant_level['prob']

    # ==========================================
    # MATRIX 1: Default Threshold (0.50)
    # ==========================================
    y_pred_default = (y_probs >= 0.50).astype(int)
    plot_cm(
        y_true, 
        y_pred_default, 
        title="Confusion Matrix (Default Threshold 0.50)", 
        filename="4.5_confusion_matrix_default.png"
    )

    # ==========================================
    # MATRIX 2: Optimal Threshold 
    # ==========================================
    y_pred_optimal = (y_probs >= OPTIMAL_THRESHOLD).astype(int)
    plot_cm(
        y_true, 
        y_pred_optimal, 
        title=f"Confusion Matrix (Tuned Threshold {OPTIMAL_THRESHOLD})", 
        filename="4.5_confusion_matrix_optimal.png"
    )

    print("\n🎉 Done! Images are in the 'report_images' folder.")

if __name__ == "__main__":
    main()