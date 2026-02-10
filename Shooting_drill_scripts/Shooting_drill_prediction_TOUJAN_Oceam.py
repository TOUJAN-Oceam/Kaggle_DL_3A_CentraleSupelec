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
import matplotlib.pyplot as plt
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

# --- IMPORT FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from usefull_scripts.get_device import get_device
from usefull_scripts.set_seed import set_seed

device = get_device()


HP = {
    "seeds": [42, 2024, 777, 99, 123], 
    "n_folds": 5,
    
    "input_dim": 10,
    "hidden_dim": 416,
    
    "batch_size": 16,
    "epochs": 60,
    "learning_rate": 0.00028,
    "weight_decay": 2.76e-6,
    "noise_level": 0.0153,
    "dropout_rate": 0.296,
    
    "sched_factor": 0.88,
    "sched_patience": 3,
    "sched_threshold": 0.00076
}

# --- 2. DATA & MODEL DEFINITIONS ---

class ShootingDataset(Dataset):
    def __init__(self, features, targets=None, noise_level=0.0):
        super().__init__()
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

class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.layer2 = nn.Linear(config['hidden_dim'], 1)
        self.output_act = nn.Softplus()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.output_act(x)
        return x

# --- 3. DATA LOADING ---

print("Loading data...")
train_path = '../10-minute-shooting-drill/train.csv'
test_path = '../10-minute-shooting-drill/test.csv'

# Classifier
binary_pred_path = 'stacking/pred_binary_classifier.csv'

df_train_full = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


try:
    df_binary = pd.read_csv(binary_pred_path)
    binary_probs_test = df_binary['prob_0'].values
except FileNotFoundError:
    binary_probs_test = None

feature_cols = df_train_full.drop(columns=['id', 'Number Of Crossbars']).columns.tolist() 

X = df_train_full[feature_cols].values
y = df_train_full['Number Of Crossbars'].values 
X_kaggle_test = df_test[feature_cols].values

# --- 4. TRAINING PIPELINE ---

final_test_preds = np.zeros(len(X_kaggle_test))


global_cv_scores = []
all_history_losses = [] 
all_true_values = []
all_pred_values = []

## Just to get some time
start_time = time.time()

for seed_idx, seed in enumerate(HP['seeds']):
    print(f"\n🌱 Random seed: {seed} ({seed_idx+1}/{len(HP['seeds'])})")
    set_seed(seed)
    
    kf = KFold(n_splits=HP['n_folds'], shuffle=True, random_state=seed)
    seed_test_preds = np.zeros(len(X_kaggle_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        y_train_log = np.log1p(y_train)

        train_dataset = ShootingDataset(X_train, y_train_log, noise_level=HP['noise_level'])
        val_dataset = ShootingDataset(X_val, y_val, noise_level=0.0)
        
        train_loader = DataLoader(train_dataset, batch_size=HP['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=HP['batch_size'], shuffle=False)
        
        model = NeuralNetwork(HP).to(device)
        criterion = nn.L1Loss()
        
        optimizer = optim.AdamW(model.parameters(), lr=HP['learning_rate'], weight_decay=HP['weight_decay'])
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=HP['sched_factor'], 
            patience=HP['sched_patience'], 
            threshold=HP['sched_threshold'], 
            min_lr=1e-7
        )
        
        best_val_mae = float('inf')
        best_model_state = None
        fold_losses = []
        
        # Training Loop
        for epoch in range(HP['epochs']):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets) 
                loss.backward()
                optimizer.step()
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # Inverse log transform
                    real_preds = torch.expm1(model(inputs)) 
                    val_loss += criterion(real_preds, targets).item()
            
            val_mae = val_loss / len(val_loader)
            fold_losses.append(val_mae)
            scheduler.step(val_mae)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = copy.deepcopy(model.state_dict())
        
        plt.figure(figsize=(10, 5))
        plt.plot(fold_losses, label='Validation MAE', color='orange')
        plt.title(f'Convergence Validation - Seed {seed} - Fold {fold+1}')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        os.makedirs('shooting_drill_training_plots', exist_ok=True)
        
        # Sauvegarder l'image
        plt.savefig(f'shooting_drill_training_plots/val_mae_seed{seed}_fold{fold+1}.png')
        plt.close()

        print(f"   Fold {fold+1}: Best MAE = {best_val_mae:.4f}")
        global_cv_scores.append(best_val_mae)
        all_history_losses.append(fold_losses)
        
        model.load_state_dict(best_model_state)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                real_preds = torch.expm1(model(inputs))
                all_true_values.extend(targets.cpu().numpy().flatten())
                all_pred_values.extend(real_preds.cpu().numpy().flatten())
        
        # TEST TIME AUGMENTATION (TTA) 
        X_test_scaled = scaler.transform(X_kaggle_test)
        
        loader_clean = DataLoader(ShootingDataset(X_test_scaled, noise_level=0.0), batch_size=HP['batch_size'])
        preds_clean = []
        with torch.no_grad():
            for inputs in loader_clean:
                preds_clean.extend(torch.expm1(model(inputs.to(device))).cpu().numpy().flatten())
                
        loader_noise1 = DataLoader(ShootingDataset(X_test_scaled, noise_level=0.015), batch_size=HP['batch_size'])
        preds_noise1 = []
        with torch.no_grad():
            for inputs in loader_noise1:
                preds_noise1.extend(torch.expm1(model(inputs.to(device))).cpu().numpy().flatten())
        
        loader_noise2 = DataLoader(ShootingDataset(X_test_scaled, noise_level=0.025), batch_size=HP['batch_size'])
        preds_noise2 = []
        with torch.no_grad():
            for inputs in loader_noise2:
                preds_noise2.extend(torch.expm1(model(inputs.to(device))).cpu().numpy().flatten())

        avg_preds = (np.array(preds_clean) + np.array(preds_noise1) + np.array(preds_noise2)) / 3.0
        seed_test_preds += avg_preds / HP['n_folds']

    final_test_preds += seed_test_preds / len(HP['seeds'])
    print(f"✅ Seed {seed} completed.")

# --- 5. REPORTING, POST-PROCESSING & SAVING ---

elapsed = time.time() - start_time
mean_score = np.mean(global_cv_scores)

print("\n" + "="*40)
print(f"⏱️  Total Time: {elapsed:.2f} s")
print(f"📊 GLOBAL SCORE (Simplicity + TTA): {mean_score:.4f}")
print("="*40)


final_preds_processed = final_test_preds.copy()

if binary_probs_test is not None:
    print("🔧 Application du filtre binaire (Two-Stage)...")
    
    THRESHOLD_ZERO = 0.85 
    
    mask_zero = binary_probs_test > THRESHOLD_ZERO
    n_zeros_forced = np.sum(mask_zero)
    
    final_preds_processed[mask_zero] = 0.0
    
    print(f"   👉 {n_zeros_forced} prédictions forcées à 0 (Seuil confiance: {THRESHOLD_ZERO})")

plt.figure(figsize=(10, 5))
plt.hist(final_test_preds, bins=50, alpha=0.5, label='Original Regression')
plt.hist(final_preds_processed, bins=50, alpha=0.5, label='With Binary Filter')
plt.legend()
plt.title("Impact du Filtre Binaire sur la Distribution")
plt.show()

# Save Original
sub_orig = pd.DataFrame({'id': df_test['id'], 'Number Of Crossbars': final_test_preds})
sub_orig.to_csv(f"submission_Simplicity_TTA_{mean_score:.4f}.csv", index=False)


# Save Processed
sub_proc = pd.DataFrame({'id': df_test['id'], 'Number Of Crossbars': final_preds_processed})
filename_proc = f"submission_{mean_score:.4f}.csv"
sub_proc.to_csv(filename_proc, index=False)
print(f"📝 Fichier filtré '{filename_proc}' prêt.")