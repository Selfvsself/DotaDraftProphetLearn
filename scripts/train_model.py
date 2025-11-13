import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_classes.mlp_dota_model_medium import DotaModel
import torch.nn.functional as F


class DotaDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_parquet(data_path)

        radiant_ids = np.stack(df['radiant_team'].values)
        self.radiant_ids = torch.tensor(radiant_ids, dtype=torch.long)

        radiant_attrs = np.stack(df['radiant_primary_attributes'].values)
        self.radiant_attrs = torch.tensor(radiant_attrs, dtype=torch.long)

        radiant_type_attack = np.stack(df['radiant_attack_type'].values)
        self.radiant_type_attack = torch.tensor(radiant_type_attack, dtype=torch.long)

        radiant_roles = np.array([[hero_roles for hero_roles in match] for match in df['radiant_roles'].values],
                                 dtype=np.long)
        self.radiant_roles = torch.tensor(radiant_roles, dtype=torch.long)

        radiant_winrate = np.stack(df['radiant_winrate'].values)
        self.radiant_winrate = torch.tensor(radiant_winrate, dtype=torch.float32)

        radiant_pickrate = np.stack(df['radiant_pickrate'].values)
        self.radiant_pickrate = torch.tensor(radiant_pickrate, dtype=torch.float32)

        dire_ids = np.stack(df['dire_team'].values)
        self.dire_ids = torch.tensor(dire_ids, dtype=torch.long)

        dire_attrs = np.stack(df['dire_primary_attributes'].values)
        self.dire_attrs = torch.tensor(dire_attrs, dtype=torch.long)

        dire_type_attack = np.stack(df['dire_attack_type'].values)
        self.dire_type_attack = torch.tensor(dire_type_attack, dtype=torch.long)

        dire_roles = np.array([[hero_roles for hero_roles in match] for match in df['dire_roles'].values],
                              dtype=np.int64)
        self.dire_roles = torch.tensor(dire_roles, dtype=torch.long)

        dire_winrate = np.stack(df['dire_winrate'].values)
        self.dire_winrate = torch.tensor(dire_winrate, dtype=torch.float32)

        dire_pickrate = np.stack(df['dire_pickrate'].values)
        self.dire_pickrate = torch.tensor(dire_pickrate, dtype=torch.float32)

        self.avg_rank_tier = torch.tensor(df['avg_rank_tier'].values, dtype=torch.float32)
        self.duration = torch.tensor(df['duration'].values, dtype=torch.float32)
        self.y = torch.tensor(df['radiant_win'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'radiant_ids': self.radiant_ids[idx],
            'dire_ids': self.dire_ids[idx],
            'radiant_attrs': self.radiant_attrs[idx],
            'dire_attrs': self.dire_attrs[idx],
            'radiant_type_attack': self.radiant_type_attack[idx],
            'dire_type_attack': self.dire_type_attack[idx],
            'radiant_roles': self.radiant_roles[idx],
            'dire_roles': self.dire_roles[idx],
            'radiant_winrate': self.radiant_winrate[idx],
            'dire_winrate': self.dire_winrate[idx],
            'radiant_pickrate': self.radiant_pickrate[idx],
            'dire_pickrate': self.dire_pickrate[idx],
            'avg_rank_tier': self.avg_rank_tier[idx],
            'duration': self.duration[idx],
            'y': self.y[idx]
        }


OUTPUT_TRAIN_DIR = f'../train/{time.strftime("%Y%m%d-%H%M%S")}'
os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)

LOG_FILENAME = os.path.join(OUTPUT_TRAIN_DIR, 'train.log')
DATASET_DIR = '../data/processed'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logging.info(f"Output directory: {OUTPUT_TRAIN_DIR}")

BATCH_SIZE = 128
LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.0001
NUM_EPOCHS = 15

logging.info(f"Batch size: {BATCH_SIZE}")
logging.info(f"Learning rate: {LEARNING_RATE}")
logging.info(f"L2 regularization: {L2_REGULARIZATION}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

train_dataset = DotaDataset(os.path.join(DATASET_DIR, 'train.parquet'))
val_dataset = DotaDataset(os.path.join(DATASET_DIR, 'validation.parquet'))

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = DotaModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
criterion = nn.BCEWithLogitsLoss()

with open(os.path.join(OUTPUT_TRAIN_DIR, "model_architecture.txt"), "w") as f:
    print(model, file=f)

logging.info("Start training...")

VAL_ACC_MAX = 0
plt_train_losses = []
plt_train_accs = []
plt_val_losses = []
plt_val_accs = []
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct_train, total_train = 0, 0, 0
    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False)
    for batch in train_loop:
        radiant_ids = batch.get('radiant_ids').to(device)
        dire_ids = batch.get('dire_ids').to(device)
        avg_rank_tiers = batch.get('avg_rank_tier').to(device)
        durations = batch.get('duration').to(device)
        y_batch = batch.get('y').to(device)

        optimizer.zero_grad()
        y_pred = model(
            radiant_ids,
            dire_ids,
            avg_rank_tiers,
            durations)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (torch.sigmoid(y_pred) > 0.5).float()
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)

        with torch.no_grad():
            model.hero_emb.weight.data = F.normalize(model.hero_emb.weight.data, p=2, dim=1)

        train_loop.set_postfix(avg_loss=total_loss / (train_loop.n + 1), acc=correct_train / total_train)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    model.eval()
    val_loss, correct, total = 0, 0, 0
    val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False)
    with torch.no_grad():
        for batch in val_loop:
            radiant_ids = batch.get('radiant_ids').to(device)
            dire_ids = batch.get('dire_ids').to(device)
            avg_rank_tiers = batch.get('avg_rank_tier').to(device)
            durations = batch.get('duration').to(device)
            y_batch = batch.get('y').to(device)

            y_pred = model(
                radiant_ids,
                dire_ids,
                avg_rank_tiers,
                durations)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

            preds = (torch.sigmoid(y_pred) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            val_loop.set_postfix(avg_loss=val_loss / (val_loop.n + 1), acc=correct / total)

    val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    plt_train_losses.append(avg_loss)
    plt_train_accs.append(train_acc)
    plt_val_losses.append(val_loss)
    plt_val_accs.append(val_acc)

    hero_mean_norm = model.hero_emb.weight.norm(dim=1).mean().item()
    if val_acc > VAL_ACC_MAX:
        BEST_MODEL_FILENAME = f'model_{epoch + 1}_{avg_loss:.4f}_{train_acc:.3f}_' \
                              f'{val_loss:.4f}_{val_acc:.3f}.pth'
        torch.save(model.state_dict(), os.path.join(OUTPUT_TRAIN_DIR, BEST_MODEL_FILENAME))
        VAL_ACC_MAX = val_acc
        logging.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.3f} | "
            f"Mean embedding: {hero_mean_norm:.3f} | "
            f"Model saved: {BEST_MODEL_FILENAME}"
        )
    else:
        logging.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.3f} | "
            f"Mean embedding: {hero_mean_norm:.3f}"
        )

torch.save(model.state_dict(), os.path.join(OUTPUT_TRAIN_DIR, "model_final.pth"))

plt.figure(figsize=(8, 6))
plt.plot(plt_train_losses, label='Train Loss')
plt.plot(plt_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_TRAIN_DIR, 'loss.png'))
plt.close()

# Accuracy
plt.figure(figsize=(8, 6))
plt.plot(plt_train_accs, label='Train Accuracy')
plt.plot(plt_val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_TRAIN_DIR, 'accuracy.png'))
plt.close()
