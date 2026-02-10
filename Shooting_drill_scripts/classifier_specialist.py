import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
import copy
import sys
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from usefull_scripts.get_device import get_device
from usefull_scripts.set_seed import set_seed

device = get_device()

HP = {
    "seeds": [42, 2024, 777, 99, 123],
    "n_folds": 5,
    "input_dim": 10,
    "hidden_dim_1": 416,
    "hidden_dim_2": 128,
    "output_dim": 2,
    "batch_size": 32,
    "epochs": 60,
    "lr": 0.0005,
    "wd": 1e-4,
    "noise": 0.025,
    "dropout": 0.2,
    "label_smoothing": 0.1,
    "sched_factor": 0.5,
    "sched_patience": 5,
}

class BinaryDataset(Dataset):
    def __init__(self, X, y=None, noise=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor((y > 0).astype(int), dtype=torch.long) if y is not None else None
        self.noise = noise

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.noise > 0: x += torch.randn_like(x) * self.noise
        return (x, self.y[idx]) if self.y is not None else x

class Classifier(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.fc1 = nn.Linear(hp['input_dim'], hp['hidden_dim_1'])
        self.bn1 = nn.BatchNorm1d(hp['hidden_dim_1'])
        self.fc2 = nn.Linear(hp['hidden_dim_1'], hp['hidden_dim_2'])
        self.bn2 = nn.BatchNorm1d(hp['hidden_dim_2'])
        self.head = nn.Linear(hp['hidden_dim_2'], hp['output_dim'])
        
        self.act = nn.SiLU()
        self.drop = nn.Dropout(hp['dropout'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.head(x)
        return x

def run_epoch(model, loader, optimizer=None, criterion=None):
    if optimizer: model.train()
    else: model.eval()
    
    total_loss, correct, n, probs = 0, 0, 0, []
    
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            x = batch[0] if isinstance(batch, list) else batch
            y = batch[1] if isinstance(batch, list) else None
            out = model(x.to(device))
            
            if y is not None:
                loss = criterion(out, y.to(device))
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1) == y.to(device)).sum().item()
                n += y.size(0)
            
            if not optimizer:
                probs.append(F.softmax(out, dim=1).cpu().numpy())
                
    if not optimizer and n == 0: return np.concatenate(probs), 0, 0
    return (np.concatenate(probs) if probs else None), total_loss / len(loader), correct / n

print("Loading data...")
# Adapte les chemins ici si besoin
path_train = r'C:\Users\black\Documents\centraleSupelec\3A\cours\DL&NLP\Kaggle_ML\10-minute-shooting-drill\train.csv'
path_test = r'C:\Users\black\Documents\centraleSupelec\3A\cours\DL&NLP\Kaggle_ML\10-minute-shooting-drill\test.csv'

df_tr, df_te = pd.read_csv(path_train), pd.read_csv(path_test)
cols = df_tr.drop(columns=['id', 'Number Of Crossbars']).columns.tolist()
X, y = df_tr[cols].values, df_tr['Number Of Crossbars'].values
X_te = df_te[cols].values

oof_probs = np.zeros((len(X), 2))
test_probs = np.zeros((len(X_te), 2))

for seed in HP['seeds']:
    print(f"Seed: {seed}")
    set_seed(seed)
    kf = KFold(n_splits=HP['n_folds'], shuffle=True, random_state=seed)
    
    for tr_idx, val_idx in kf.split(X):
        # Weights handling
        y_bin = (y[tr_idx] > 0).astype(int)
        counts = np.bincount(y_bin)
        w = torch.tensor(len(y_bin) / (len(counts) * counts), dtype=torch.float32).to(device)
        
        # Preprocessing
        scaler = StandardScaler()
        X_tr, X_val = scaler.fit_transform(X[tr_idx]), scaler.transform(X[val_idx])
        
        dl_tr = DataLoader(BinaryDataset(X_tr, y[tr_idx], HP['noise']), batch_size=HP['batch_size'], shuffle=True)
        dl_val = DataLoader(BinaryDataset(X_val, y[val_idx]), batch_size=HP['batch_size'])
        
        model = Classifier(HP).to(device)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=HP['label_smoothing'])
        opt = optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=HP['wd'])
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=HP['sched_factor'], patience=HP['sched_patience'])
        
        best_loss, best_w = float('inf'), None
        for _ in range(HP['epochs']):
            run_epoch(model, dl_tr, opt, crit)
            _, val_loss, _ = run_epoch(model, dl_val, None, crit)
            sched.step(val_loss)
            if val_loss < best_loss: best_loss, best_w = val_loss, copy.deepcopy(model.state_dict())
        
        model.load_state_dict(best_w)
        oof_probs[val_idx] += run_epoch(model, dl_val, None, crit)[0] / len(HP['seeds'])
        
        # TTA
        X_te_sc = scaler.transform(X_te)
        p1 = run_epoch(model, DataLoader(BinaryDataset(X_te_sc), batch_size=HP['batch_size']))[0]
        p2 = run_epoch(model, DataLoader(BinaryDataset(X_te_sc, noise=0.02), batch_size=HP['batch_size']))[0]
        test_probs += ((p1 + p2) / 2) / (HP['n_folds'] * len(HP['seeds']))

os.makedirs("stacking", exist_ok=True)
pd.DataFrame(oof_probs, columns=['prob_0', 'prob_pos']).assign(id=df_tr['id']).to_csv("stacking/oof_binary_classifier.csv", index=False)
pd.DataFrame(test_probs, columns=['prob_0', 'prob_pos']).assign(id=df_te['id']).to_csv("stacking/pred_binary_classifier.csv", index=False)
print("Done.")