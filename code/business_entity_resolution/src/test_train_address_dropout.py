import os
import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

FEATURE_COLS = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_compact_match', 'name_acronym_match', 'is_addr_missing',
    'street_num_match', 'street_name_sim', 'city_state_sim',
    'addr_token_sort', 'digits_match', 'country_match',
    'is_dba_pattern', 'source_origin'
]

COL_DTYPES = {
    'name_ratio': np.float32,
    'name_token_sort': np.float32,
    'name_token_set': np.float32,
    'name_partial': np.float32,
    'name_compact_match': np.float32,
    'name_acronym_match': np.float32,
    'is_addr_missing': np.float32,
    'street_num_match': np.float32,
    'street_name_sim': np.float32,
    'city_state_sim': np.float32,
    'addr_token_sort': np.float32,
    'digits_match': np.float32,
    'country_match': np.float32,
    'is_dba_pattern': np.float32,
    'source_origin': np.float32,
    'label': np.int8
}

ADDR_COL_INDICES = [
    FEATURE_COLS.index('is_addr_missing'),
    FEATURE_COLS.index('street_num_match'),
    FEATURE_COLS.index('street_name_sim'),
    FEATURE_COLS.index('city_state_sim'),
    FEATURE_COLS.index('addr_token_sort'),
    FEATURE_COLS.index('digits_match'),
    FEATURE_COLS.index('is_dba_pattern')
]

def load_with_address_dropout(csv_path, dropout_rate=0.15, is_train=True):
    print(f"Loading {csv_path} (address_dropout={dropout_rate if is_train else 0.0})...", flush=True)
    t0 = time.time()
    X_chunks = []
    y_chunks = []
    chunk_size = 500000
    
    np.random.seed(42)
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=FEATURE_COLS + ['label'], dtype=COL_DTYPES):
        X_mat = chunk[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_vec = chunk['label'].to_numpy(dtype=np.int8)
        
        if is_train and dropout_rate > 0:
            # Apply address dropout to a random subset
            mask = np.random.rand(len(X_mat)) < dropout_rate
            if np.any(mask):
                X_mat[mask, FEATURE_COLS.index('is_addr_missing')] = 1.0
                X_mat[mask, FEATURE_COLS.index('street_num_match')] = 0.5
                X_mat[mask, FEATURE_COLS.index('street_name_sim')] = 0.5
                X_mat[mask, FEATURE_COLS.index('city_state_sim')] = 0.5
                X_mat[mask, FEATURE_COLS.index('addr_token_sort')] = 0.0
                X_mat[mask, FEATURE_COLS.index('digits_match')] = 0.0
                X_mat[mask, FEATURE_COLS.index('is_dba_pattern')] = 0.0
                
        X_chunks.append(X_mat)
        y_chunks.append(y_vec)
        
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    print(f"Loaded {len(X):,} rows in {time.time()-t0:.1f}s.", flush=True)
    return X, y

if __name__ == '__main__':
    train_csv = 'output/full_train_features_v3.csv'
    val_csv = 'output/full_val_features_v3.csv'
    
    X_train, y_train = load_with_address_dropout(train_csv, dropout_rate=0.15, is_train=True)
    X_val, y_val = load_with_address_dropout(val_csv, dropout_rate=0.0, is_train=False)
    
    print("\nConstructing LightGBM datasets...", flush=True)
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=True)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data, free_raw_data=False)
    del X_train, y_train
    gc.collect()
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': 10,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'min_child_samples': 40,
        'verbose': -1,
        'n_jobs': 6
    }
    
    print("\nTraining LightGBM with 15% Address Dropout (800 trees)...", flush=True)
    t0 = time.time()
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=800,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f"Training finished in {time.time()-t0:.1f}s!", flush=True)
    
    out_model = 'output/lgbm_model_v4_dropout.txt'
    booster.save_model(out_model)
    print(f"Saved model to {out_model}", flush=True)
    
    print("\nPredicting on validation set...", flush=True)
    val_probs = booster.predict(X_val)
    np.save('output/val_probs_lgb_v4_dropout.npy', val_probs)
    print(f"Saved validation probabilities to output/val_probs_lgb_v4_dropout.npy", flush=True)
    
    # Feature importance
    print("\nFeature Importance (Split Gain):", flush=True)
    imp = booster.feature_importance(importance_type='gain')
    for feat, score in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        print(f"  {feat:<22}: {score:>14.1f}", flush=True)
