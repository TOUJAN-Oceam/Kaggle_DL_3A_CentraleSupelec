import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from usefull_scripts.get_device import get_device
from usefull_scripts.set_seed import set_seed

# --- CONFIGURATION ---
DEVICE = get_device()
N_TRIALS = 300  
N_FOLDS = 5     
EPOCHS = 25     

try:
    df_train = pd.read_csv('train.csv')
except:
    df_train = pd.read_csv(r'MonChemin\gymnastic-exam\train.csv')

feature_cols = [
    'age', 'backflip_quality', 'eyebrow_length', 'teeth_whiteness',
    'ear_size', 'frontflip_quality', 'stamina', 'shoulder_width'
]
X = df_train[feature_cols].values
y = df_train['passes_exam'].values

# --- UTILITAIRES ---
class GymDataset(Dataset):
    def __init__(self, features, targets=None, noise_level=0.0):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1) if targets is not None else None
        self.noise_level = noise_level 

    def __len__(self): 
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
        if self.targets is not None:
            return x, self.targets[idx]
        return x

def get_model(trial, input_dim):
    n_layers = trial.suggest_int('n_layers', 1, 2)
    layers = []
    in_features = input_dim
    
    activation_name = trial.suggest_categorical("activation", ["ReLU", "SiLU", "GELU"])
    activation_fn = getattr(nn, activation_name)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)

    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 32, 1024, step=32)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(activation_fn())
        layers.append(nn.Dropout(dropout_rate))
        in_features = out_features
    
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers).to(DEVICE)

def objective(trial):
    # --- 1. ARCHITECTURE ---
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_sizes = []
    for i in range(n_layers):
        layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 32, 512, step=32)) 

    # --- 2. HYPERPARAMETRES ---
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64]) 
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    noise_level = trial.suggest_float("noise_level", 0.0, 0.05)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    activation_name = trial.suggest_categorical("activation", ["ReLU", "SiLU", "GELU"])

    # --- 3. SCHEDULER ---
    scheduler_name = trial.suggest_categorical("scheduler", ["Cosine", "Plateau"])
    sched_t0 = trial.suggest_int("t0_scheduler", 5, 20) if scheduler_name == "Cosine" else 0
    sched_patience = trial.suggest_int("sched_patience", 3, 8) if scheduler_name == "Plateau" else 0
    sched_factor = trial.suggest_float("sched_factor", 0.1, 0.8) if scheduler_name == "Plateau" else 0.0

    # --- 4. BOUCLE CROSS-VALIDATION ROBUSTE ---
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_accuracies = [] 
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]
        
        train_dl = DataLoader(GymDataset(X_tr, y_tr, noise_level=noise_level), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(GymDataset(X_va, y_va, noise_level=0.0), batch_size=batch_size, shuffle=False)
        
        layers = []
        in_f = len(feature_cols)
        act_fn = getattr(nn, activation_name)
        for h_dim in layer_sizes:
            layers.append(nn.Linear(in_f, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_f = h_dim
        layers.append(nn.Linear(in_f, 1))
        model = nn.Sequential(*layers).to(DEVICE)
        
        criterion = nn.BCEWithLogitsLoss()
        
        if optimizer_name == "Adam": opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamW": opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else: opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if scheduler_name == "Cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=sched_t0, T_mult=2)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=sched_factor, patience=sched_patience)

        best_fold_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            for x, t in train_dl:
                opt.zero_grad()
                loss = criterion(model(x.to(DEVICE)), t.to(DEVICE))
                loss.backward()
                opt.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, t in val_dl:
                    logits = model(x.to(DEVICE))
                    val_loss += criterion(logits, t.to(DEVICE)).item()
                    # Calcul Accuracy
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == t.to(DEVICE)).sum().item()
                    total += t.size(0)
            
            avg_val_loss = val_loss / len(val_dl)
            current_acc = correct / total

            if current_acc > best_fold_acc:
                best_fold_acc = current_acc

            # Step Scheduler
            if scheduler_name == "Cosine": scheduler.step()
            else: scheduler.step(avg_val_loss)

            # Pruning sur le Fold 0 uniquement (pour gagner du temps)
            if fold == 0:
                trial.report(current_acc, epoch)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()

        # On ajoute le MEILLEUR score atteint sur ce fold (pas forcément le dernier)
        fold_accuracies.append(best_fold_acc)

    # On retourne la moyenne des meilleurs scores de chaque fold
    return np.mean(fold_accuracies)

# --- LANCEMENT ---
if __name__ == "__main__":
    print(f"🚀 Lancement Optuna avec CV intégrée ({N_TRIALS} essais)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print("\n🏆 MEILLEURS HYPERPARAMÈTRES (Score CV Moyen) :")
    print(study.best_params)
    print(f"🌟 Meilleure Accuracy Moyenne : {study.best_value:.4f}")
    
    # Graphiques
    try:
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
    except: pass