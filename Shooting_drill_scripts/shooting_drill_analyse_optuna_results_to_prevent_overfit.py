import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import random
import os
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from usefull_scripts.get_device import get_device
from usefull_scripts.set_seed import set_seed

# --- CONFIGURATION ---
SEEDS = [42, 2024, 777, 99, 123]  
DEVICE = get_device()
EPOCHS = 20  
THRESHOLD_VALUE = 1.2 
THRESHOLD_LOW = 1.1    

try:
    df_train = pd.read_csv('train.csv')
    df_optuna = pd.read_csv('optuna_secure_results.csv') 
except:
    path = r'C:\Users\black\Documents\centraleSupelec\3A\cours\DL&NLP\Kaggle_ML\10-minute-shooting-drill'
    try:
        df_train = pd.read_csv(os.path.join(path, 'train.csv'))
        df_optuna = pd.read_csv(os.path.join(path, 'optuna_secure_results.csv'))
    except FileNotFoundError:
        print("⚠️ Fichiers non trouvés. Vérifie les chemins.")
        # On crée des dataframes vides pour que le code compile (à retirer en prod)
        df_train = pd.DataFrame(columns=['id', 'Number Of Crossbars', 'col1'])
        df_optuna = pd.DataFrame(columns=['state', 'value', 'number'])

# Préparation Data
if not df_train.empty:
    feature_cols = df_train.drop(columns=['id', 'Number Of Crossbars']).columns.tolist()
    X_raw = df_train[feature_cols].values
    y_raw = df_train['Number Of Crossbars'].values
else:
    X_raw, y_raw = [], []

if not df_optuna.empty:
    candidates = df_optuna[
        (df_optuna['state'] == 'COMPLETE') & 
        (df_optuna['value'] < THRESHOLD_VALUE) &
        (df_optuna['value'] > THRESHOLD_LOW) # <-- AJOUT ICI
    ].sort_values(by='value')

    print(f"🎯 {len(candidates)} Candidats trouvés entre {THRESHOLD_LOW} et {THRESHOLD_VALUE}.")
    print("🚀 Démarrage du Benchmark Rigoureux (Real MAE)...")
else:
    candidates = pd.DataFrame()
    print("⚠️ Aucun résultat Optuna chargé.")

# --- CLASSES UTILITAIRES ---
class ShootingDataset(Dataset):
    def __init__(self, features, targets=None, noise_level=0.0):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1) if targets is not None else None
        self.noise_level = noise_level
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = self.features[idx]
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        if self.targets is not None:
            return x, self.targets[idx]
        return x

def build_model_from_params(params, device):
    layers = []
    in_features = 10
    n_layers = int(params['params_n_layers'])
    dropout = params['params_dropout_rate']
    
    for i in range(n_layers):
        # Récupération dynamique du nombre d'unités pour la couche i
        units_key = f'params_n_units_l{i}'
        out_features = int(params[units_key])
        
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))
        in_features = out_features
        
    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Softplus())
    
    return nn.Sequential(*layers).to(device)

results_log = []

if not candidates.empty:

    for idx, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Benchmarking"):
        trial_number = row['number']
        optuna_score = row['value']
        
        # Paramètres
        batch_size = int(row['params_batch_size'])
        lr = row['params_lr']
        weight_decay = row['params_weight_decay']
        noise_level = row['params_noise_level']
        optimizer_name = row['params_optimizer']
        
        # Scheduler Params
        patience = int(row['params_patience'])
        factor = row['params_factor']
        threshold = row['params_threshold']
        
        seed_scores = []
        
        for seed in SEEDS:
            set_seed(seed)
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X_raw):
                X_tr, X_val = X_raw[train_idx], X_raw[val_idx]
                y_tr, y_val = y_raw[train_idx], y_raw[val_idx]
                
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)
                y_tr_log = np.log1p(y_tr)
                
                train_dl = DataLoader(ShootingDataset(X_tr, y_tr_log, noise_level), batch_size=batch_size, shuffle=True)
                val_dl = DataLoader(ShootingDataset(X_val, y_val, 0.0), batch_size=batch_size, shuffle=False)
                
                model = build_model_from_params(row, DEVICE)
                
                # Optimizer
                if optimizer_name == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                    
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=factor, patience=patience, threshold=threshold, min_lr=1e-7
                )
                criterion = nn.L1Loss()
                
                best_mae_real = float('inf')
                
                for epoch in range(EPOCHS):
                    model.train()
                    for x_b, y_b in train_dl:
                        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                        optimizer.zero_grad()
                        pred = model(x_b)
                        loss = criterion(pred, y_b)
                        loss.backward()
                        optimizer.step()
                    
                    model.eval()
                    val_loss_real = 0.0
                    with torch.no_grad():
                        for x_b, y_b in val_dl:
                            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                            real_pred = torch.expm1(model(x_b)) 
                            val_loss_real += criterion(real_pred, y_b).item()
                    
                    mae_real = val_loss_real / len(val_dl)
                    scheduler.step(mae_real)
                    
                    if mae_real < best_mae_real:
                        best_mae_real = mae_real
                
                fold_scores.append(best_mae_real)
            
            seed_scores.append(np.mean(fold_scores))
        
        mean_real_mae = np.mean(seed_scores)
        std_real_mae = np.std(seed_scores)
        
        results_log.append({
            'Trial': int(trial_number),
            'Optuna_Val': optuna_score,
            'Real_MAE_Mean': mean_real_mae,
            'Real_MAE_Std': std_real_mae,
            'n_layers': int(row['params_n_layers']),
            'lr': lr,
            'dropout': row['params_dropout_rate'],
            'params_full': row.to_dict() 
        })
        
        print(f"   Trial {int(trial_number)}: Real MAE = {mean_real_mae:.4f} (Optuna said {optuna_score:.4f})")

    df_results = pd.DataFrame(results_log)
    df_results = df_results.sort_values(by='Real_MAE_Mean')

    csv_name = "benchmark_results_final.csv"
    df_results.to_csv(csv_name, index=False)

    print("\n" + "="*50)
    print("🏆 TOP 5 MODÈLES (VERIFIÉS)")
    print("="*50)
    print(df_results[['Trial', 'Real_MAE_Mean', 'Real_MAE_Std', 'Optuna_Val', 'n_layers', 'lr']].head(5))

    # Graphiques
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_results, x='Optuna_Val', y='Real_MAE_Mean', hue='n_layers', palette='viridis', size='Real_MAE_Std')
    plt.title("Réalité vs Optuna (Bulles = Instabilité)")
    plt.xlabel("Score Optuna (Log-based)")
    plt.ylabel("Vrai Score MAE (Validé)")
    plt.grid(True, alpha=0.3)

    # Plot 2: Top 10 Barplot
    plt.subplot(1, 2, 2)
    top_10 = df_results.head(10)
    sns.barplot(data=top_10, x='Real_MAE_Mean', y='Trial', orient='h', hue='Trial', legend=False)
    plt.title("Top 10 Meilleurs Essais")
    plt.xlabel("MAE Moyenne (5 Seeds)")
    plt.xlim(top_10['Real_MAE_Mean'].min() - 0.02, top_10['Real_MAE_Mean'].max() + 0.02)

    plt.tight_layout()
    plt.show() # Affiche et sauvegarde
    plt.savefig("benchmark_summary.png")

    print(f"✅ Analyse terminée. Résultats détaillés dans '{csv_name}'")