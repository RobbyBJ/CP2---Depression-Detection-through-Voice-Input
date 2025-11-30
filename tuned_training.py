import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
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
# UPDATE THIS PATH to your new big dataset
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv" 
TEST_SIZE = 0.20  
RANDOM_STATE = 42 
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def run_training_pipeline():
    print("🚀 LOADING SEGMENT DATASET...")
    df = pd.read_csv(INPUT_CSV)
    
    # --- CRITICAL CHANGE: GROUP AWARE SPLIT ---
    # We cannot just split rows randomly. We must split by Participant_ID.
    
    # 1. Get unique participants and their labels
    participant_data = df[['participant_id', 'PHQ8_Binary']].drop_duplicates()
    
    # 2. Split the PARTICIPANTS, not the segments
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(
        participant_data['participant_id'], 
        test_size=TEST_SIZE, 
        stratify=participant_data['PHQ8_Binary'], # Keep ratio of depressed/healthy balanced
        random_state=RANDOM_STATE
    )
    
    print(f"   Participants in Train: {len(train_ids)} | Participants in Test: {len(test_ids)}")

    # 3. Create the actual Train/Test DataFrames based on those IDs
    train_df = df[df['participant_id'].isin(train_ids)]
    test_df = df[df['participant_id'].isin(test_ids)]
    
    # 4. Separate X and y
    # Note: We drop participant_id NOW, after splitting
    X_train = train_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_train = train_df['PHQ8_Binary']
    
    X_test = test_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_test = test_df['PHQ8_Binary']

    print(f"   Train Segments: {len(X_train)} | Test Segments: {len(X_test)}")
    print(f"   Train Class Balance: {y_train.value_counts().to_dict()}")

    # ---------------------------------------------------------

    models_config = {
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE, class_weight='balanced'),
            # REDUCED GRID: SVM on 20k rows is slow. Removed 'poly' and some C values.
            'params': {
                'classifier__C': [0.1, 1, 10], 
                'classifier__kernel': ['rbf'], 
                'classifier__gamma': ['scale']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [5, 10]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            # KNN is slow with large data, kept n_neighbors low
            'params': {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['distance']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'classifier__var_smoothing': [1e-9, 1e-8]
            }
        }
    }

    # 4. TRAINING LOOP
    results = []
    
    print("\n⚔️ STARTING MODEL COMPARISON ON SEGMENTS...")
    
    for name, config in models_config.items():
        print(f"\n... Training {name} ...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()), 
            ('classifier', config['model'])
        ])
        
        # Standard StratifiedKFold is okay here because we have huge data,
        # but technically GroupKFold is better. For simplicity/speed, we use StratifiedKFold
        # on the segments. The REAL test is the X_test we created above.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        grid = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=cv, 
            scoring='f1', 
            n_jobs=-1, 
            verbose=1
        )
        
        # This might take a while for SVM!
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        
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
            'Best Params': grid.best_params_
        })
        
        print(f"   Best Params: {grid.best_params_}")
        print(f"   Test F1-Score: {f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1-Score', ascending=False)
    
    print("\n🏆 FINAL RESULTS (Segment Level) 🏆")
    print("="*60)
    print(results_df[['Model', 'Accuracy', 'F1-Score', 'Recall (Sensitivity)', 'Specificity']])
    print("="*60)
    
    results_df.to_csv("model_results.csv", index=False)
    print("\n✅ Results saved to 'model_results.csv'")

if __name__ == "__main__":
    run_training_pipeline()