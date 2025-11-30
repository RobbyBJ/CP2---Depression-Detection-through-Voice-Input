import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# --- IMPORT MODELS ---
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ================= CONFIGURATION =================
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def run_baseline_training():
    print("🚀 LOADING SEGMENT DATASET (BASELINE RUN)...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. GROUP-AWARE SPLIT (CRITICAL)
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
    X_test = test_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_test = test_df['PHQ8_Binary']

    print(f"   Train Segments: {len(X_train)} | Test Segments: {len(X_test)}")

    # 2. DEFINE BASELINE MODELS (DEFAULTS ONLY)
    # We add class_weight='balanced' only because your dataset is imbalanced.
    # Everything else is LEFT AS DEFAULT to establish a baseline.
    models_config = {
        'SVM': SVC(random_state=RANDOM_STATE, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000),
        'KNN': KNeighborsClassifier(), # Default is 5 neighbors
        'Naive Bayes': GaussianNB()
    }

    results = []
    
    print("\n⚔️ STARTING BASELINE TRAINING (NO TUNING)...")
    
    for name, model in models_config.items():
        print(f"\n... Training {name} ...")
        
        # Simple Pipeline: Impute -> Scale -> Train
        # NO GridSearchCV here. Just .fit()
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()), 
            ('classifier', model)
        ])
        
        # Train once on the full training set
        pipeline.fit(X_train, y_train)
        
        # Predict on Test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        spec = calculate_specificity(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        recall = report['1']['recall'] 
        precision = report['1']['precision']

        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1-Score': f1,
            'Recall (Sensitivity)': recall,
            'Precision': precision,
            'Specificity': spec,
            'Configuration': 'Baseline (Default)'
        })
        
        print(f"   F1-Score: {f1:.4f}")

    # 3. SAVE RESULTS
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1-Score', ascending=False)
    
    print("\n🏆 BASELINE RESULTS 🏆")
    print(results_df[['Model', 'Accuracy', 'F1-Score', 'Recall (Sensitivity)']])
    
    results_df.to_csv("baseline_model_results.csv", index=False)
    print("\n✅ Baseline results saved to 'baseline_model_results.csv'")

if __name__ == "__main__":
    run_baseline_training()