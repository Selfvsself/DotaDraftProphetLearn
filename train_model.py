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


class DotaDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_parquet(data_path)

        radiant_ids = np.stack(df['radiant_team'].values)
        self.radiant_ids = torch.tensor(radiant_ids, dtype=torch.long)

        dire_ids = np.stack(df['dire_team'].values)
        self.dire_ids = torch.tensor(dire_ids, dtype=torch.long)
        self.avg_rank_tier = torch.tensor(df['avg_rank_tier'].values, dtype=torch.float32)
        self.num_rank_tier = torch.tensor(df['num_rank_tier'].values, dtype=torch.float32)
        self.duration = torch.tensor(df['duration'].values, dtype=torch.float32)
        self.y = torch.tensor(df['radiant_win'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.radiant_ids[idx], \
            self.dire_ids[idx], \
            self.avg_rank_tier[idx], \
            self.num_rank_tier[idx], \
            self.duration[idx], \
            self.y[idx]


class DotaModel(nn.Module):
    def __init__(self, num_heroes, emb_dim=32, dropout_p=0.3):
        super(DotaModel, self).__init__()
        self.hero_emb = nn.Embedding(
            num_embeddings=num_heroes,
            embedding_dim=emb_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * emb_dim + 3, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1)
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, radiant_ids, dire_ids, avg_rank_tiers, num_rank_tiers, durations):
        emb_r = self.hero_emb(radiant_ids)
        emb_d = self.hero_emb(dire_ids)
        team_r = emb_r.mean(dim=1)
        team_d = emb_d.mean(dim=1)
        x = torch.cat([team_r,
                       team_d,
                       avg_rank_tiers.unsqueeze(1),
                       num_rank_tiers.unsqueeze(1),
                       durations.unsqueeze(1)],
                      dim=1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit.squeeze(1)


output_dir = f'train/{time.strftime("%Y%m%d-%H%M%S")}'
os.makedirs(output_dir, exist_ok=True)

LOG_FILENAME = os.path.join(output_dir, 'train.log')
DATASET_DIR = 'datasets/'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logging.info(f"Output directory: {output_dir}")

BATCH_SIZE = 128
NUM_HEROES = 126
LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.0001
NUM_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

train_dataset = DotaDataset(os.path.join(DATASET_DIR, 'train.parquet'))
val_dataset = DotaDataset(os.path.join(DATASET_DIR, 'validation.parquet'))

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = DotaModel(NUM_HEROES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
criterion = nn.BCEWithLogitsLoss()

# save model architecture
with open(os.path.join(output_dir, "model_architecture.txt"), "w") as f:
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
    for radiant_ids, dire_ids, avg_rank_tiers, num_rank_tiers, durations, y_batch in train_loop:
        radiant_ids = radiant_ids.to(device)
        dire_ids = dire_ids.to(device)
        avg_rank_tiers = avg_rank_tiers.to(device)
        num_rank_tiers = num_rank_tiers.to(device)
        durations = durations.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(radiant_ids, dire_ids, avg_rank_tiers, num_rank_tiers, durations)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (torch.sigmoid(y_pred) > 0.5).float()
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)

        train_loop.set_postfix(avg_loss=total_loss / (train_loop.n + 1), acc=correct_train / total_train)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    model.eval()
    val_loss, correct, total = 0, 0, 0
    val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False)
    with torch.no_grad():
        for radiant_ids, dire_ids, avg_rank_tiers, num_rank_tiers, durations, y_batch in val_loop:
            radiant_ids = radiant_ids.to(device)
            dire_ids = dire_ids.to(device)
            avg_rank_tiers = avg_rank_tiers.to(device)
            num_rank_tiers = num_rank_tiers.to(device)
            durations = durations.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(radiant_ids, dire_ids, avg_rank_tiers, num_rank_tiers, durations)
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

    if val_acc > VAL_ACC_MAX:
        BEST_MODEL_FILENAME = f'model_{epoch + 1}_{avg_loss:.4f}_{train_acc:.3f}_' \
                              f'{val_loss / len(val_loader):.4f}_{val_acc:.3f}.pth'
        torch.save(model.state_dict(), os.path.join(output_dir, BEST_MODEL_FILENAME))
        VAL_ACC_MAX = val_acc
        logging.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.3f} | "
            f"Model saved: {BEST_MODEL_FILENAME}"
        )
    else:
        logging.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.3f}"
        )

torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pth"))

plt.figure(figsize=(8, 6))
plt.plot(plt_train_losses, label='Train Loss')
plt.plot(plt_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss.png'))
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
plt.savefig(os.path.join(output_dir, 'accuracy.png'))
plt.close()
