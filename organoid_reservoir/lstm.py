"""Small CPU LSTM baselines (optional; needs the ``lstm`` extra, i.e. PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class LSTMConfig:
    """Training settings for the LSTM baselines."""

    enabled: bool = True
    hidden: int = 64
    epochs: int = 150
    lr: float = 0.01
    weight_decay: float = 1e-4
    window: int = 20
    """Input window length for the time-series LSTM."""
    threads: int = 4


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _make_model(n_in: int, hidden: int, n_out: int) -> Any:
    import torch
    from torch import nn

    class Net(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x: Any, lengths: Any = None) -> Any:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                _, (h, _) = self.lstm(packed)
            else:
                _, (h, _) = self.lstm(x)
            return self.head(h[-1])

    del torch
    return Net()


def lstm_classify(
    X_train: list[FloatArray],
    y_train: NDArray[np.int64],
    X_test: list[FloatArray],
    config: LSTMConfig,
    seed: int = 0,
) -> NDArray[np.int64]:
    """Train an LSTM sequence classifier on z-scored inputs; return test predictions."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(config.threads)
    allf = np.concatenate(X_train)
    mu, sd = allf.mean(0), allf.std(0) + 1e-8

    def batch(seqs: list[FloatArray]) -> tuple[Any, Any]:
        lengths = torch.tensor([len(s) for s in seqs])
        padded = torch.zeros(len(seqs), int(lengths.max()), seqs[0].shape[1])
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = torch.tensor((s - mu) / sd, dtype=torch.float32)
        return padded, lengths

    xtr, ltr = batch(X_train)
    xte, lte = batch(X_test)
    ytr = torch.tensor(y_train)
    model = _make_model(xtr.shape[2], config.hidden, int(y_train.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(config.epochs):
        perm = torch.randperm(len(ytr), generator=gen)
        for idx in perm.split(64):
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx], ltr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte, lte).argmax(1).numpy()
    return np.asarray(pred, dtype=np.int64)


def lstm_forecast(
    u: FloatArray,
    y: FloatArray,
    n_washout: int,
    n_train: int,
    config: LSTMConfig,
    seed: int = 0,
) -> FloatArray:
    """Train an LSTM mapping the last ``window`` inputs to the target.

    Returns predictions for every time step after ``n_washout + n_train``.
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(config.threads)
    w = config.window
    mu, sd = u[: n_washout + n_train].mean(), u[: n_washout + n_train].std() + 1e-8
    ymu, ysd = y[: n_washout + n_train].mean(), y[: n_washout + n_train].std() + 1e-8
    z = (u - mu) / sd
    idx = np.arange(max(w - 1, n_washout), len(u))
    windows = np.stack([z[i - w + 1 : i + 1] for i in idx])[:, :, None]
    targets = (y[idx] - ymu) / ysd
    split = n_washout + n_train
    train = idx < split
    xtr = torch.tensor(windows[train], dtype=torch.float32)
    ytr = torch.tensor(targets[train], dtype=torch.float32)[:, None]
    xte = torch.tensor(windows[~train], dtype=torch.float32)
    model = _make_model(1, config.hidden, 1)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(config.epochs):
        perm = torch.randperm(len(ytr), generator=gen)
        for b in perm.split(256):
            opt.zero_grad()
            loss = torch.mean((model(xtr[b]) - ytr[b]) ** 2)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xte).numpy()[:, 0]
    return np.asarray(pred * ysd + ymu, dtype=np.float64)
