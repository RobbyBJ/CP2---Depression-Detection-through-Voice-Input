import os
import pandas as pd
import numpy as np
import opensmile
import audiofile
from tqdm import tqdm

# ================= CONFIGURATION =================
# Point to your V4 Augmented/Balanced Data
PROCESSED_AUDIO_DIR = r"C:\Users\User\Desktop\processed_balanced" 

TRAIN_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

OUTPUT_TRAIN = r"C:\Users\User\Desktop\depression_train_opensmile.csv"
OUTPUT_TEST = r"C:\Users\User\Desktop\depression_test_opensmile.csv"

# Logic Settings (Same as V8)
MAX_TEST_SEGMENTS = 50 
MIN_SEGMENTS = 5
# =================================================

def extract_opensmile_features():
    print("🚀 Initializing OpenSMILE (eGeMAPS - The Standard for Depression)...")
    
    # Initialize OpenSMILE with eGeMAPS feature set (Standard for Affect/Depression)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    return smile

def process_split(smile, label_file, output_csv, split_name):
    print(f"\n📘 Processing {split_name} split...")
    
    if not os.path.exists(label_file):
        print(f"❌ Label file not found: {label_file}")
        return

    labels = pd.read_csv(label_file)
    if 'Participant_ID' in labels.columns:
        labels.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    
    labels['participant_id'] = labels['participant_id'].astype(int)
    label_map = pd.Series(labels.PHQ8_Binary.values, index=labels.participant_id).to_dict()

    all_features = []

    for root, _, files in os.walk(PROCESSED_AUDIO_DIR):
        folder_name = os.path.basename(root)
        if not folder_name.isdigit(): continue

        pid = int(folder_name)
        if pid not in label_map: continue

        target_label = label_map[pid]
        all_wavs = [f for f in files if f.endswith('.wav')]
        
        # --- SAME LOGIC AS V8 (Smart Sampling) ---
        if split_name == "TRAIN":
            final_files = all_wavs # Keep Augmentations
        else:
            final_files = [f for f in all_wavs if "_aug" not in f] # No Augmentations
            if len(final_files) > MAX_TEST_SEGMENTS:
                final_files = np.random.choice(final_files, MAX_TEST_SEGMENTS, replace=False)
        
        # Looping Logic
        if len(final_files) < MIN_SEGMENTS:
             if len(final_files) == 0: continue
             needed = MIN_SEGMENTS - len(final_files)
             extras = np.random.choice(final_files, needed, replace=True)
             final_files = np.concatenate([final_files, extras])

        print(f"➡️ Participant {pid} ({len(final_files)} segments)")

        for wav in tqdm(final_files, leave=False):
            full_path = os.path.join(root, wav)
            try:
                # Extract Features using OpenSMILE
                # It returns a DataFrame with 1 row and 88 columns
                df_feat = smile.process_file(full_path)
                
                # Reset index to get a clean row
                df_feat = df_feat.reset_index(drop=True)
                
                # Add Metadata
                df_feat['participant_id'] = pid
                df_feat['PHQ8_Binary'] = target_label
                
                all_features.append(df_feat)
                
            except Exception as e:
                pass # Skip bad files

    if all_features:
        # Combine all rows
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"✅ {split_name} Saved: {len(final_df)} segments | {final_df.shape[1]} features")
    else:
        print(f"⚠️ No data extracted for {split_name}")

def main():
    smile = extract_opensmile_features()
    
    # Process Train (Augmented)
    process_split(smile, TRAIN_LABELS, OUTPUT_TRAIN, "TRAIN")
    
    # Process Test (Balanced)
    process_split(smile, DEV_LABELS, OUTPUT_TEST, "TEST")
    
    print("\n🎉 DONE! You now have 'Paper-Quality' features.")

if __name__ == "__main__":
    main()