import logging

import numpy as np
import torch
import pandas as pd

from ncf import NCF, _NeuMF, LR, WEIGHT_DECAY, EPOCHS, EARLY_STOPPING
from base_model import _rmse, TARGET
from preprocess import load_trust_graph

logger = logging.getLogger(__name__)

SOCIAL_LAMBDA = 0.01


class SocialNCF(NCF):
    name = "social_ncf"

    def __init__(self):
        super().__init__()
        self._dataset = None

    def evaluate(self, dataset, train, val, test, rebuild=False):
        self._dataset = dataset
        return super().evaluate(dataset, train, val, test, rebuild=rebuild)

    def _social_reg(self, edges: torch.Tensor) -> torch.Tensor:
        src_emb = self._model.user_mf(edges[:, 0])
        dst_emb = self._model.user_mf(edges[:, 1])
        return ((src_emb - dst_emb) ** 2).sum(dim=-1).mean()

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self._global_mean = float(train[TARGET].mean())
        self._n_users = train["user_code"].max() + 1
        self._n_items = train["item_code"].max() + 1

        self._model = _NeuMF(self._n_users, self._n_items).to(self._device)
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        loss_fn = torch.nn.MSELoss()

        self._trust_df = load_trust_graph(self._dataset)
        reg_edges = None
        if self._trust_df is not None and not self._trust_df.empty:
            known_mask = (self._trust_df["src"] < self._n_users) & (
                self._trust_df["dst"] < self._n_users
            )
            known_edges = self._trust_df[known_mask]
            if not known_edges.empty:
                reg_edges = torch.tensor(
                    known_edges[["src", "dst"]].values, dtype=torch.long
                ).to(self._device)
            logger.info(
                f"  Trust graph: {len(self._trust_df):,} total edges, "
                f"{len(known_edges):,} between known users"
            )

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
                if reg_edges is not None:
                    loss = loss + SOCIAL_LAMBDA * self._social_reg(reg_edges)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(users)
            train_loss /= len(train)

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

    def _cold_neighbor_map(self) -> dict[int, list[int]]:
        """Map cold user codes to their list of known (train) neighbor codes."""
        cold = self._trust_df[
            (self._trust_df["src"] >= self._n_users)
            & (self._trust_df["dst"] < self._n_users)
        ]
        return cold.groupby("src")["dst"].apply(list).to_dict()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        known = (
            (df["user_code"] >= 0)
            & (df["item_code"] >= 0)
            & (df["user_code"] < self._n_users)
            & (df["item_code"] < self._n_items)
        )
        cold_user = (
            (df["user_code"] >= self._n_users)
            & (df["item_code"] >= 0)
            & (df["item_code"] < self._n_items)
        )

        preds = np.full(len(df), self._global_mean)

        if known.any():
            loader = self._make_loader(df[known])
            preds[known.values] = self._predict_loader(loader)

        if cold_user.any() and self._trust_df is not None and not self._trust_df.empty:
            neighbor_map = self._cold_neighbor_map()
            cold_df = df[cold_user]

            self._model.eval()
            with torch.no_grad():
                mf_weights = self._model.user_mf.weight.data
                mlp_weights = self._model.user_mlp.weight.data

                for _, group in cold_df.groupby("user_code"):
                    user_code = int(group["user_code"].iloc[0])
                    friends = neighbor_map.get(user_code, [])
                    if not friends:
                        continue

                    friend_codes = torch.tensor(
                        friends, dtype=torch.long, device=self._device
                    )
                    avg_mf = mf_weights[friend_codes].mean(dim=0)
                    avg_mlp = mlp_weights[friend_codes].mean(dim=0)

                    items = torch.tensor(
                        group["item_code"].values, dtype=torch.long, device=self._device
                    )
                    mf_out = avg_mf.unsqueeze(0) * self._model.item_mf(items)
                    mlp_in = torch.cat(
                        [
                            avg_mlp.unsqueeze(0).expand(len(items), -1),
                            self._model.item_mlp(items),
                        ],
                        dim=-1,
                    )
                    mlp_out = self._model.mlp(mlp_in)
                    out = self._model.predict_layer(
                        torch.cat([mf_out, mlp_out], dim=-1)
                    )
                    preds[group.index.values] = out.squeeze(-1).cpu().numpy()

        return preds
