import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ================= CONFIGURATION =================
TRAIN_CSV = r"C:\Users\User\Desktop\depression_train_opensmile.csv"
MODEL_OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\baseline_model_opensmile"
RANDOM_STATE = 42
TOP_K_FEATURES = 40 
# =================================================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def run_training():
    print("🚀 LOADING TRAIN DATA (Segment-Level)...")
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Error: {TRAIN_CSV} not found.")
        return

    df_train = pd.read_csv(TRAIN_CSV)
    
    # Drop metadata for training
    X_train = df_train.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df_train['PHQ8_Binary']

    print(f"✅ Training on {len(X_train)} segments.")
    print(f"   Class Balance: {y_train.value_counts().to_dict()}")

    # ================= DEFINE MODELS =================
    # No SMOTE, No Class Weights (Data Volume handles the bias)
    models_config = {
        'SVM': SVC(random_state=RANDOM_STATE, probability=True),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, max_depth=10, n_jobs=-1),
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        'KNN': KNeighborsClassifier(n_neighbors=9),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, 
                                 tree_method='hist', eval_metric='logloss', random_state=RANDOM_STATE)
    }

    print(f"\n⚔️ STARTING TRAINING...")

    for name, model in models_config.items():
        print(f"\n🧩 Training {name}...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)),
            ('classifier', model)
        ])

        try:
            pipeline.fit(X_train, y_train)
            
            # Save Model
            save_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_v8.pkl")
            joblib.dump(pipeline, save_path)
            print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Failed {name}: {e}")

    print("\n🎉 Training Complete. Now run the Testing Script.")

if __name__ == "__main__":
    run_training()