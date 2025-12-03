import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ================= CONFIGURATION =================
# Ensure these 3 files exist in your folder
PHASE1_FILE = "baseline_model_results.csv"      # From run_baseline_training.py
PHASE2_FILE = "tuned_model_results.csv"               # From tuned_training.py
PHASE3_FILE = "final_patient_diagnosis.csv"     # From majority_voting.py
# =================================================

# Set a professional style
sns.set_theme(style="whitegrid")

def plot_phase1_baseline():
    """Generates Figure 4.1: Baseline Performance (The Accuracy Paradox)"""
    print("📊 Generating Figure 4.1: Baseline Performance...")
    try:
        df = pd.read_csv(PHASE1_FILE)
        
        # Melt for Seaborn
        df_melt = df.melt(id_vars="Model", value_vars=["Accuracy", "Recall (Sensitivity)"], 
                          var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="RdBu")
        
        plt.title("Phase 1: Baseline Performance (The Accuracy Paradox)", fontsize=14, fontweight='bold')
        plt.ylabel("Score (0.0 - 1.0)")
        plt.ylim(0, 0.8)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Random Guess")
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig("Figure_4.1_Baseline_Paradox.png", dpi=300)
        print("✅ Saved Figure_4.1_Baseline_Paradox.png")
    except Exception as e:
        print(f"❌ Error Plotting Phase 1: {e}")

def plot_phase2_tuned():
    """Generates Figure 4.2: Tuned Segment Performance"""
    print("📊 Generating Figure 4.2: Tuned Segment Performance...")
    try:
        df = pd.read_csv(PHASE2_FILE)
        
        df_melt = df.melt(id_vars="Model", value_vars=["Accuracy", "Recall (Sensitivity)"], 
                          var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="viridis")
        
        plt.title("Phase 2: Tuned Segment-Level Performance", fontsize=14, fontweight='bold')
        plt.ylabel("Score (0.0 - 1.0)")
        plt.ylim(0, 0.8)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig("Figure_4.2_Tuned_Performance.png", dpi=300)
        print("✅ Saved Figure_4.2_Tuned_Performance.png")
    except Exception as e:
        print(f"❌ Error Plotting Phase 2: {e}")

def plot_phase3_confusion():
    """Generates Figure 4.3: Final Confusion Matrix"""
    print("📊 Generating Figure 4.3: Final Confusion Matrix...")
    try:
        df = pd.read_csv(PHASE3_FILE)
        
        cm = confusion_matrix(df['True_Label'], df['Final_Prediction'])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
        
        plt.title("Phase 3: Final Confusion Matrix (Threshold 0.3)", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted Label (0=Healthy, 1=Depressed)")
        plt.ylabel("True Label (0=Healthy, 1=Depressed)")
        
        plt.tight_layout()
        plt.savefig("Figure_4.3_Final_Confusion_Matrix.png", dpi=300)
        print("✅ Saved Figure_4.3_Final_Confusion_Matrix.png")
    except Exception as e:
        print(f"❌ Error Plotting Phase 3: {e}")

def plot_overall_improvement():
    """Generates Figure 4.4: Sensitivity Improvement Journey"""
    print("📊 Generating Figure 4.4: Overall Improvement...")
    
    # Manually defined based on your results history
    data = {
        'Phase': ['1. Baseline\n(Random Forest)', '2. Tuned Segments\n(Logistic Regression)', '3. Final Voting\n(Logistic Regression)'],
        'Sensitivity': [0.13, 0.49, 0.56]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['Phase'], df['Sensitivity'], color=['#e74c3c', '#f1c40f', '#2ecc71'])
    
    plt.title("Sensitivity Improvement Across Project Phases", fontsize=14, fontweight='bold')
    plt.ylabel("Sensitivity (Recall)")
    plt.ylim(0, 0.7)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=12, weight='bold')
        
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Figure_4.4_Sensitivity_Improvement.png", dpi=300)
    print("✅ Saved Figure_4.4_Sensitivity_Improvement.png")

if __name__ == "__main__":
    plot_phase1_baseline()
    plot_phase2_tuned()
    plot_phase3_confusion()
    plot_overall_improvement()
    print("\n🎉 All plots generated successfully!")