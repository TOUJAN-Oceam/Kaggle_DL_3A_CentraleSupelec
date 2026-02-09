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
import copy
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from usefull_scripts.get_device import get_device
from usefull_scripts.set_seed import set_seed

HP = {
    "seeds": [42, 2024, 777, 99, 123],
    "n_folds": 5,
    "input_dim": 8,
    "batch_size": 16,
    "epochs": 150,  
    "learning_rate": 0.003046, 
    "weight_decay": 0.000357,
    "dropout_rate": 0.192,
    "noise_level": 0.01, 
    "t0_scheduler": 14
}

DEVICE = get_device()

class GymDataset(Dataset):
    def __init__(self, features, targets=None, noise_level=0.0):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1) if targets is not None else None
        self.noise_level = noise_level

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.noise_level > 0:
            x += torch.randn_like(x) * self.noise_level
        if self.targets is not None:
            return x, self.targets[idx]
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, hp):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(hp['input_dim'], 160)
        self.bn1 = nn.BatchNorm1d(160)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(hp['dropout_rate'])
        
        self.layer2 = nn.Linear(160, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(hp['dropout_rate'])
        
        self.output = nn.Linear(80, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.output(x)
        return x

print("Loading data...")
df_train = pd.read_csv('../gymnastic-exam/train.csv')
df_test = pd.read_csv('../gymnastic-exam/test.csv')

cols = ['age', 'backflip_quality', 'eyebrow_length', 'teeth_whiteness', 'ear_size', 'frontflip_quality', 'stamina', 'shoulder_width']
X = df_train[cols].values
y = df_train['passes_exam'].values
X_test = df_test[cols].values

final_test_preds = np.zeros(len(X_test))
final_oof_preds = np.zeros(len(X))

for seed_idx, seed in enumerate(HP['seeds']):
    print(f"\n🌱 Random seed: {seed} ({seed_idx+1}/{len(HP['seeds'])})")
    set_seed(seed)
    
    kf = KFold(n_splits=HP['n_folds'], shuffle=True, random_state=seed)
    seed_preds = np.zeros(len(X_test))
    
    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        
        train_dl = DataLoader(GymDataset(X_tr, y[train_idx], HP['noise_level']), batch_size=HP['batch_size'], shuffle=True)
        val_dl = DataLoader(GymDataset(X_val, y[val_idx], 0.0), batch_size=HP['batch_size'], shuffle=False)
        
        model = NeuralNetwork(HP).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=HP['learning_rate'], weight_decay=HP['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=HP['t0_scheduler'], T_mult=2)
        criterion = nn.BCEWithLogitsLoss()
        train_losses = []
        val_losses = []

        best_loss, best_weights = float('inf'), None
        
        for epoch in range(HP['epochs']):
            model.train()
            running_train_loss = 0.0
            
            for x_b, t_b in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(x_b.to(DEVICE)), t_b.to(DEVICE))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            
            avg_train_loss = running_train_loss / len(train_dl)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_b, t_b in val_dl:
                    val_loss += criterion(model(x_b.to(DEVICE)), t_b.to(DEVICE)).item()
            
            avg_val_loss = val_loss / len(val_dl)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step()
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = copy.deepcopy(model.state_dict())
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Loss Convergence - Seed {seed} - Fold {val_idx[0]}') # ID unique
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend()
        plt.grid(True)
        
        # Créer un dossier pour ranger les images si besoin
        os.makedirs('gymnastic_loss_plots', exist_ok=True)
        plt.savefig(f'gymnastic_loss_plots/loss_seed{seed}_fold_{val_idx[0]}.png')
        plt.close() 

        model.load_state_dict(best_weights)

        
        with torch.no_grad():
            oof_p = []
            for x_b, _ in val_dl:
                oof_p.extend(torch.sigmoid(model(x_b.to(DEVICE))).cpu().numpy().flatten())
            final_oof_preds[val_idx] += np.array(oof_p) / len(HP['seeds'])
            
        X_test_sc = scaler.transform(X_test)
        loaders = [
            DataLoader(GymDataset(X_test_sc, targets=None, noise_level=0.0), batch_size=HP['batch_size']),
            DataLoader(GymDataset(X_test_sc, targets=None, noise_level=HP['noise_level']), batch_size=HP['batch_size']),
            DataLoader(GymDataset(X_test_sc, targets=None, noise_level=HP['noise_level']*1.5), batch_size=HP['batch_size'])
        ]
        
        fold_preds = np.zeros(len(X_test))
        with torch.no_grad():
            for loader in loaders:
                preds = []
                for x_b in loader:
                    preds.extend(torch.sigmoid(model(x_b.to(DEVICE))).cpu().numpy().flatten())
                fold_preds += np.array(preds)
        
        seed_preds += fold_preds / 3 / HP['n_folds']

    final_test_preds += seed_preds / len(HP['seeds'])



acc = np.mean((final_oof_preds > 0.5) == y)
print(f"Accuracy: {acc:.4f}")

os.makedirs('stacking', exist_ok=True)
pd.DataFrame({'id': df_train['id'], 'pred': final_oof_preds}).to_csv('stacking/oof_nn_robust.csv', index=False)
pd.DataFrame({'id': df_test['id'], 'passes_exam': (final_test_preds > 0.5).astype(int)}).to_csv(f'submission_robust_{acc:.4f}.csv', index=False)
