import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ================= CONFIGURATION =================
TRAIN_CSV = r"C:\Users\User\Desktop\depression_train_dataset.csv"
MODEL_OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\tuned_models"
RANDOM_STATE = 42

# Define the "Search Space" for each model
# The grid search will try all combinations to find the best one for SENSITIVITY.
MODEL_PARAMS = {
    'SVM': {
        'model': SVC(probability=True, random_state=RANDOM_STATE),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf'],
            'classifier__gamma': ['scale', 0.1],
            'selector__k': [30, 40, 50]  # Let the model choose best feature count
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_leaf': [1, 2, 4],
            'selector__k': [30, 40, 50]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        'params': {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'selector__k': [30, 40, 50]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [5, 7, 9, 13],
            'classifier__weights': ['uniform', 'distance'],
            'selector__k': [30, 40, 50]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=RANDOM_STATE),
        'params': {
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__n_estimators': [100, 200],
            'selector__k': [30, 40, 50]
        }
    }
}
# =================================================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def run_tuning():
    print("🚀 LOADING TRAIN DATA (Segment-Level)...")
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Error: {TRAIN_CSV} not found.")
        return

    df_train = pd.read_csv(TRAIN_CSV)
    
    # Drop metadata for training
    X_train = df_train.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df_train['PHQ8_Binary']

    print(f"✅ Tuning on {len(X_train)} segments.")
    print(f"   Class Balance: {y_train.value_counts().to_dict()}")

    print("\n⚔️ STARTING HYPERPARAMETER TUNING (Optimizing for RECALL/SENSITIVITY)...")

    # Use Stratified K-Fold to maintain class balance in validation splits
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, config in MODEL_PARAMS.items():
        print(f"\n🧩 Tuning {name}...")
        
        # 1. Define Pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif)), # k is tuned in grid
            ('classifier', config['model'])
        ])

        # 2. Setup Grid Search
        # scoring='recall' forces the model to pick params that minimize False Negatives
        grid = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=cv, 
            scoring='recall', 
            n_jobs=-1, 
            verbose=1
        )

        try:
            # 3. Run Tuning
            grid.fit(X_train, y_train)
            
            # 4. Report Results
            print(f"   🏆 Best Params: {grid.best_params_}")
            print(f"   🏆 Best Recall Score (CV): {grid.best_score_:.2%}")
            
            # 5. Save Best Model
            save_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_tuned_v8.pkl")
            joblib.dump(grid.best_estimator_, save_path)
            print(f"   ✅ Saved: {save_path}")

        except Exception as e:
            print(f"   ❌ Failed {name}: {e}")

    print("\n🎉 Tuning Complete. Now run your 'test_v8_voting_final.py' on these new models!")

if __name__ == "__main__":
    run_tuning()