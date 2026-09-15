"""Train the neural beta model on the full CRSP common-stock panel.

Reproduces the architecture and tuned hyperparameters from neural-beta.ipynb
(128 hidden units, ReLU, lr 3e-4, 12-month lookback, batch 256, 30 epochs) and
writes the test-period betas to a CSV keyed by firm-month, ready for run_bab.py.

Two differences from the notebook, both deliberate:

Features are built once over the whole panel and the samples are then split by
the year of the month being predicted. Building each split separately, as the
notebook does, throws away the first `lookback` months of every firm in every
split, which silently removes most of 2018 from the test set.

The epoch with the best validation RMSE is restored at the end instead of
keeping whatever the thirtieth epoch happened to give.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import bab
import run_bab

SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralBetaMLP(nn.Module):
    """One hidden layer mapping a feature window to a single beta.

    The output layer stays linear: betas can be negative, and a squashing
    activation there would rule that out. `alpha` is one scalar shared across
    the panel, exactly as in the notebook.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, activation: str = "relu"):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.activation = {
            "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(), "linear": None,
        }[activation]
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.layer1(x)
        if self.activation is not None:
            h = self.activation(h)
        return self.layer2(h)

    def loss(self, beta_hat, mkt_next, r_next):
        r_hat = self.alpha + beta_hat * mkt_next
        return torch.sqrt(torch.mean((r_next.view_as(r_hat) - r_hat) ** 2))


def run_epoch(model, loader, optimizer=None):
    """One pass. Returns RMSE pooled over samples, not averaged over batches.

    Averaging per-batch RMSEs would understate the loss whenever batches differ
    in size, and the last batch of an epoch almost always does.
    """
    train = optimizer is not None
    model.train(train)
    sq_err, n = 0.0, 0

    with torch.set_grad_enabled(train):
        for x, r, m in loader:
            x, r, m = x.to(DEVICE), r.to(DEVICE), m.to(DEVICE)
            beta_hat = model(x)
            loss = model.loss(beta_hat, m, r)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sq_err += float(loss.detach()) ** 2 * len(x)
            n += len(x)

    return float(np.sqrt(sq_err / n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msf", default="MSF_dl.csv")
    ap.add_argument("--out", default="neural_betas.csv")
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-end", type=int, default=2012)
    ap.add_argument("--val-end", type=int, default=2017)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--start", type=int, default=2005)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"device: {DEVICE}")
    df = run_bab.load_crsp(args.msf, args.start, args.test_end,
                           lookback_years=args.lookback // 12 + 1)
    df = df.drop_nulls(["RETX", "vwretd"]).sort(["PERMNO", "year", "month"])

    industries = sorted(df["industry"].unique().to_list())
    df = df.with_columns([
        (pl.col("industry") == ind).cast(pl.Int8).alias(f"industry_{ind}")
        for ind in industries
    ])
    print(f"panel: {df.height:,} rows, {df['PERMNO'].n_unique():,} firms, "
          f"{len(industries)} industries")

    X, r_next, mkt_next, keys = bab.build_keyed_arrays(df, lookback=args.lookback)
    print(f"samples: {len(X):,}  features: {X.shape[1]}")

    year = keys["year"].to_numpy()
    splits = {
        "train": year <= args.train_end,
        "val": (year > args.train_end) & (year <= args.val_end),
        "test": (year > args.val_end) & (year <= args.test_end),
    }

    def loader(mask, shuffle):
        ds = TensorDataset(torch.from_numpy(X[mask]),
                           torch.from_numpy(r_next[mask]).unsqueeze(1),
                           torch.from_numpy(mkt_next[mask]).unsqueeze(1))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    for name, mask in splits.items():
        print(f"  {name}: {mask.sum():,} samples")

    train_loader = loader(splits["train"], True)
    val_loader = loader(splits["val"], False)
    test_loader = loader(splits["test"], False)

    model = NeuralBetaMLP(X.shape[1], args.hidden, args.activation).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, opt)
        va = run_epoch(model, val_loader)
        flag = ""
        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())
            flag = "  *"
        print(f"epoch {epoch:03d}  train {tr:.6f}  val {va:.6f}{flag}")

    model.load_state_dict(best_state)
    print(f"\nrestored epoch {best_epoch} (val RMSE {best_val:.6f})")
    print(f"test RMSE: {run_epoch(model, test_loader):.6f}")
    print(f"learned alpha: {float(model.alpha.detach()):.6f}")

    # Predict in panel order so the betas line up with their keys.
    model.eval()
    with torch.no_grad():
        test_X = torch.from_numpy(X[splits["test"]]).to(DEVICE)
        betas = model(test_X).squeeze(1).cpu().numpy()

    out = keys[splits["test"]].copy()
    out["beta"] = betas
    out.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}: {len(out):,} firm-months")
    print(out["beta"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
