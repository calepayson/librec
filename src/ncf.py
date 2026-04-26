import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from base_model import BaseModel, _rmse, TARGET

logger = logging.getLogger(__name__)

EMBEDDING_SIZE = 32
MLP_HIDDEN = [64, 32]
DROPOUT = 0.1
LR = 0.001
BATCH_SIZE = 8192
EPOCHS = 50
EARLY_STOPPING = 5


class _NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        # GMF path
        self.user_mf = nn.Embedding(n_users, EMBEDDING_SIZE)
        self.item_mf = nn.Embedding(n_items, EMBEDDING_SIZE)

        # MLP path
        self.user_mlp = nn.Embedding(n_users, EMBEDDING_SIZE)
        self.item_mlp = nn.Embedding(n_items, EMBEDDING_SIZE)

        mlp_sizes = [2 * EMBEDDING_SIZE] + MLP_HIDDEN
        layers = []
        for i in range(len(mlp_sizes) - 1):
            layers.append(nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
        self.mlp = nn.Sequential(*layers)

        # Combine
        self.predict_layer = nn.Linear(EMBEDDING_SIZE + MLP_HIDDEN[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user, item):
        # GMF
        mf_out = self.user_mf(user) * self.item_mf(item)

        # MLP
        mlp_in = torch.cat([self.user_mlp(user), self.item_mlp(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        out = self.predict_layer(torch.cat([mf_out, mlp_out], dim=-1))
        return out.squeeze(-1)


class NCF(BaseModel):
    name = "ncf"

    def __init__(self):
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._global_mean = None

    def _make_loader(self, df: pd.DataFrame, shuffle: bool = False) -> DataLoader:
        users = torch.tensor(df["user_code"].values, dtype=torch.long)
        items = torch.tensor(df["item_code"].values, dtype=torch.long)
        ratings = torch.tensor(df[TARGET].values, dtype=torch.float32)
        return DataLoader(
            TensorDataset(users, items, ratings),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
        )

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self._global_mean = float(train[TARGET].mean())
        self._n_users = train["user_code"].max() + 1
        self._n_items = train["item_code"].max() + 1

        self._model = _NeuMF(self._n_users, self._n_items).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        train_loader = self._make_loader(train, shuffle=True)
        val_known = val[
            (val["user_code"] >= 0)
            & (val["item_code"] >= 0)
            & (val["user_code"] < self._n_users)
            & (val["item_code"] < self._n_items)
        ]
        val_loader = self._make_loader(val_known)

        best_val_rmse = float("inf")
        patience = 0
        best_state = None

        for epoch in range(EPOCHS):
            self._model.train()
            train_loss = 0.0
            for users, items, ratings in train_loader:
                users, items, ratings = (
                    users.to(self._device),
                    items.to(self._device),
                    ratings.to(self._device),
                )
                pred = self._model(users, items)
                loss = loss_fn(pred, ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(users)
            train_loss /= len(train)

            # Validation
            val_pred = self._predict_loader(val_loader)
            val_rmse = _rmse(val_known[TARGET].to_numpy(), val_pred)

            logger.info(
                f"  Epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  val_rmse={val_rmse:.4f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
            else:
                patience += 1
                if patience >= EARLY_STOPPING:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        self._model.load_state_dict(best_state)
        logger.info(f"  Best val RMSE: {best_val_rmse:.4f}")

    def _predict_loader(self, loader: DataLoader) -> np.ndarray:
        self._model.eval()
        preds = []
        with torch.no_grad():
            for users, items, _ in loader:
                users, items = users.to(self._device), items.to(self._device)
                preds.append(self._model(users, items).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        known = (
            (df["user_code"] >= 0)
            & (df["item_code"] >= 0)
            & (df["user_code"] < self._n_users)
            & (df["item_code"] < self._n_items)
        )
        preds = np.full(len(df), self._global_mean)

        if known.any():
            known_df = df[known]
            loader = self._make_loader(known_df)
            preds[known.values] = self._predict_loader(loader)

        return preds
