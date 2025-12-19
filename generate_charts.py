import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ================= CONFIGURATION =================
# Update these paths if necessary
BASE_DIR = r"C:\Users\User\Desktop\CP2"
BASELINE_CSV = os.path.join(BASE_DIR, "baseline_results.csv")
TUNED_CSV = os.path.join(BASE_DIR, "tuned_results.csv")
ENSEMBLE_CSV = os.path.join(BASE_DIR, "ensemble_results.csv")
THRESHOLD_CSV = os.path.join(BASE_DIR, "threshold_tuning_results.csv")

OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "report_images")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})
# =================================================

def save_plot(filename):
    path = os.path.join(OUTPUT_IMG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved chart: {path}")
    plt.close()

def plot_baseline_comparison():
    print("📊 Generating Baseline Comparison Chart...")
    if not os.path.exists(BASELINE_CSV):
        print(f"⚠️ Missing: {BASELINE_CSV}")
        return

    df = pd.read_csv(BASELINE_CSV)
    # Clean model names (remove '_baseline')
    df['Model'] = df['Model'].str.replace('_baseline', '', regex=False)
    
    # Melt for grouped bar chart
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
    df_melt = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette='viridis')
    
    plt.title("Baseline Model Performance (Default Parameters)", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot("4.1_baseline_comparison.png")

def plot_tuned_impact():
    print("📊 Generating Tuned vs Baseline Chart...")
    if not os.path.exists(BASELINE_CSV) or not os.path.exists(TUNED_CSV):
        print("⚠️ Missing CSVs for comparison.")
        return

    df_base = pd.read_csv(BASELINE_CSV)
    df_tuned = pd.read_csv(TUNED_CSV)

    # Filter for F1-Score comparison
    df_base['Type'] = 'Baseline'
    df_base['Model'] = df_base['Model'].str.replace('_baseline', '')
    
    df_tuned['Type'] = 'Tuned'
    df_tuned['Model'] = df_tuned['Model'].str.replace('_tuned', '')

    # Combine
    df_combined = pd.concat([df_base[['Model', 'Sensitivity', 'Type']], df_tuned[['Model', 'Sensitivity', 'Type']]])

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_combined, x='Model', y='Sensitivity', hue='Type', palette=['#95a5a6', '#e74c3c'])
    
    plt.title("Impact of Tuning on Sensitivity (Recall)", fontsize=16, fontweight='bold')
    plt.ylabel("Sensitivity Score")
    plt.ylim(0, 1.0)
    
    # Add annotation arrows? Maybe too complex. Let's stick to bars.
    plt.tight_layout()
    save_plot("4.2_tuning_impact_sensitivity.png")

def plot_ensemble_comparison():
    print("📊 Generating Ensemble Performance Chart...")
    if not os.path.exists(ENSEMBLE_CSV):
        print(f"⚠️ Missing: {ENSEMBLE_CSV}")
        return

    df = pd.read_csv(ENSEMBLE_CSV)
    df['Model'] = df['Model'].str.replace('_ensemble', '').str.title()
    
    metrics = ['Accuracy', 'Sensitivity', 'Precision']
    df_melt = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette='rocket')
    
    plt.title("Voting vs. Stacking Ensemble (Default Threshold)", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    save_plot("4.3_ensemble_comparison.png")

def plot_threshold_curve():
    print("📊 Generating Threshold Tuning Curve...")
    if not os.path.exists(THRESHOLD_CSV):
        print(f"⚠️ Missing: {THRESHOLD_CSV}")
        return

    df = pd.read_csv(THRESHOLD_CSV)
    
    # Check if 'Model' column exists (from your updated script)
    if 'Model' in df.columns:
        # If multiple models, just pick the best one (Stacking or Voting)
        model_name = df['Model'].iloc[0] 
        df = df[df['Model'] == model_name]
        title = f"Threshold Calibration Curve ({model_name})"
    else:
        title = "Threshold Calibration Curve"

    plt.figure(figsize=(12, 7))
    
    # Plot lines
    sns.lineplot(data=df, x='Threshold', y='Sensitivity', label='Sensitivity (Recall)', linewidth=3, color='#e74c3c')
    sns.lineplot(data=df, x='Threshold', y='Precision', label='Precision', linewidth=3, color='#2ecc71')
    sns.lineplot(data=df, x='Threshold', y='Accuracy', label='Accuracy', linewidth=2, linestyle='--', color='gray')
    sns.lineplot(data=df, x='Threshold', y='Specificity', label='Specificity', linewidth=2, linestyle=':', color='blue')

    # Highlight Optimal Point (e.g., 0.42)
    # Find row with best F1
    best_row = df.loc[df['F1_Score'].idxmax()]
    plt.axvline(best_row['Threshold'], color='black', linestyle='--', alpha=0.5)
    plt.text(best_row['Threshold'], 0.2, f" Optimal Threshold: {best_row['Threshold']:.2f}", rotation=90, va='bottom')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Decision Threshold (Probability Cutoff)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    save_plot("4.4_threshold_calibration.png")

if __name__ == "__main__":
    plot_baseline_comparison()
    plot_tuned_impact()
    plot_ensemble_comparison()
    plot_threshold_curve()
    print("\n✅ All charts generated in 'report_images' folder!")