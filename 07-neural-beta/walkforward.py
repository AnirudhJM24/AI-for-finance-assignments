"""Walk-forward training of an MLP beta, refit once a year.

For each test year Y the model sees labels through December of Y-2, validates
on Y-1, and predicts Y. Nothing in a fold's fitting touches a month at or after
the year it predicts — including the feature scaler, whose mean and standard
deviation are estimated on the training fold alone. Standardising over the
whole panel is the obvious way to make the network train, and it quietly
carries future moments backwards into every earlier row.

The network maps point-in-time features to one number, beta, and the loss is
the squared error of the return it implies:

    r[t+1] = alpha + beta * mkt[t+1]

`alpha` is a single learned scalar shared across the panel. The output layer
is linear, since a beta may be negative, and its bias starts at one so the
network begins from the sensible prior that a stock moves with the market.

    python walkforward.py --msf MSF_dl.csv --out betas_pit.csv
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pit

SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BetaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        # Start every beta at one rather than at zero. A beta of zero implies
        # the stock ignores the market, which is a poor place to begin and
        # leaves the gradient on the head vanishingly small.
        nn.init.zeros_(self.head.weight)
        nn.init.ones_(self.head.bias)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.head(self.body(x))


def _epoch(model, loader, opt=None) -> float:
    """One pass, returning mean squared error pooled over samples.

    Pooling rather than averaging per-batch keeps the number comparable when
    the last batch of an epoch is short, which it almost always is.
    """
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, m in loader:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            pred = model.alpha + model(x) * m
            loss = ((y - pred) ** 2).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
    return total / n


def fit_fold(Xtr, ytr, mtr, Xva, yva, mva, *, epochs, lr, batch_size,
             hidden, dropout, patience, weight_decay, verbose=True):
    """Train one fold and return the model at its best validation epoch."""
    torch.manual_seed(SEED)

    def loader(X, y, m, shuffle):
        ds = TensorDataset(torch.from_numpy(X),
                           torch.from_numpy(y).unsqueeze(1),
                           torch.from_numpy(m).unsqueeze(1))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    model = BetaMLP(Xtr.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr_loader, va_loader = loader(Xtr, ytr, mtr, True), loader(Xva, yva, mva, False)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        tr = _epoch(model, tr_loader, opt)
        va = _epoch(model, va_loader)
        if va < best - 1e-9:
            best, best_epoch, stale = va, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if verbose:
            print(f"    epoch {epoch:02d}  train {tr:.6f}  val {va:.6f}"
                  f"{'  *' if best_epoch == epoch else ''}")
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, best, best_epoch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msf", default="MSF_dl.csv")
    ap.add_argument("--out", default="betas_pit.csv")
    ap.add_argument("--test-start", type=int, default=2018)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--hidden", default="64,32")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    np.random.seed(SEED)
    hidden = tuple(int(h) for h in args.hidden.split(","))

    print(f"device: {DEVICE}")
    panel, features = pit.build(args.msf)
    print(f"panel: {panel.height:,} usable firm-months, "
          f"{panel['PERMNO'].n_unique():,} firms, "
          f"{panel['year'].min()}-{panel['year'].max()}, "
          f"{len(features)} features")

    X, y, m, _ = pit.to_arrays(panel, features)
    year = panel["year"].to_numpy()

    out_frames = []
    for test_year in range(args.test_start, args.test_end + 1):
        # Labels land in the month after the feature row, so a feature row in
        # December of Y-1 is scored on January of Y. Training on years up to
        # Y-2 keeps every label the fit ever sees strictly before the test year.
        tr = year <= test_year - 2
        va = year == test_year - 1
        te = year == test_year
        if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
            print(f"{test_year}: not enough history, skipped")
            continue

        # Scale on the training fold only.
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True) + 1e-8
        Xs = (X - mu) / sd

        print(f"\n{test_year}: train {tr.sum():,} (to {test_year-2})  "
              f"val {va.sum():,}  test {te.sum():,}")
        model, best_val, best_epoch = fit_fold(
            Xs[tr], y[tr], m[tr], Xs[va], y[va], m[va],
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            hidden=hidden, dropout=args.dropout, patience=args.patience,
            weight_decay=args.weight_decay, verbose=not args.quiet)
        print(f"  best epoch {best_epoch}, val MSE {best_val:.6f}")

        model.eval()
        with torch.no_grad():
            betas = []
            idx = np.flatnonzero(te)
            for chunk in np.array_split(idx, max(1, len(idx) // 100_000)):
                xb = torch.from_numpy(Xs[chunk]).to(DEVICE)
                betas.append(model(xb).squeeze(1).cpu().numpy())
            betas = np.concatenate(betas)

        out_frames.append(panel[idx].select(["PERMNO", "year", "month", "t"])
                          .with_columns(pl.Series("beta_nn", betas)))
        print(f"  beta: mean {betas.mean():.3f}  sd {betas.std():.3f}  "
              f"negative {(betas < 0).mean():.2%}  "
              f"min {betas.min():.2f}  max {betas.max():.2f}")

    out = pl.concat(out_frames).sort(["t", "PERMNO"])
    out.write_csv(args.out)
    print(f"\nwrote {args.out}: {out.height:,} firm-months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
