import torch
import torch.nn as nn


class DotaModel(nn.Module):
    def __init__(self, num_heroes, emb_dim=128, dropout_p=0.5):
        super(DotaModel, self).__init__()
        self.hero_emb = nn.Embedding(
            num_embeddings=num_heroes,
            embedding_dim=emb_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * emb_dim + 3, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1)
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
