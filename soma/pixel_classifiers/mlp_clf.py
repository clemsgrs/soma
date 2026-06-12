"""Pointwise MLP per-pixel classifier (torch).

A small fully-connected net applied **independently per pixel** (``(K,) → C``) — no
spatial receptive field, so it stays inside the ``PixelClassifier`` contract (spatial
models are the neural-decoder path). It owns its own mini-batch SGD training loop and
early stopping *internally* (driven by ``X_val``/``y_val``); the soma torch ``Trainer``
and ``.pt`` fold checkpoints do not apply. Serialized as a state_dict + architecture meta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from soma.pixel_classifiers.base import PixelClassifier
from soma.pixel_classifiers.registry import pixel_classifier_registry

_MODEL_FILENAME = "model.pt"


class _PointwiseMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, num_layers) - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPixelClassifier(PixelClassifier):
    """Pointwise MLP classifier with internal epochs + early stopping.

    The input dim ``K`` is read from ``X`` at fit time (the attention/feature channel
    count), so the net is built in ``fit``. ``X_val``/``y_val`` drive early stopping on
    validation loss; ``sample_weight`` weights the per-pixel cross-entropy.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        epochs: int = 30,
        batch_size: int = 8192,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 5,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self._hidden_dim = int(hidden_dim)
        self._num_layers = int(num_layers)
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._lr = float(learning_rate)
        self._weight_decay = float(weight_decay)
        self._patience = int(patience)
        self._seed = int(seed)
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: _PointwiseMLP | None = None
        self._in_dim: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        torch.manual_seed(self._seed)
        self._in_dim = int(X.shape[1])
        model = _PointwiseMLP(self._in_dim, self._hidden_dim, self._num_layers, self._num_classes)
        model = model.to(self._device)
        opt = torch.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.long)
        wt = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            Xv = torch.as_tensor(X_val, dtype=torch.float32).to(self._device)
            yv = torch.as_tensor(y_val, dtype=torch.long).to(self._device)

        n = Xt.shape[0]
        rng = np.random.default_rng(self._seed)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_val = float("inf")
        bad_epochs = 0
        for _epoch in range(self._epochs):
            model.train()
            perm = rng.permutation(n)
            for start in range(0, n, self._batch_size):
                idx = perm[start : start + self._batch_size]
                xb = Xt[idx].to(self._device)
                yb = yt[idx].to(self._device)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb, reduction="none")
                if wt is not None:
                    wb = wt[idx].to(self._device)
                    loss = (loss * wb).sum() / wb.sum().clamp_min(1e-8)
                else:
                    loss = loss.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float(nn.functional.cross_entropy(model(Xv), yv))
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self._patience:
                        break
        if has_val:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict_proba called before fit/load.")
        self._model.eval()
        out = np.empty((X.shape[0], self._num_classes), dtype=np.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, X.shape[0], self._batch_size):
                xb = Xt[start : start + self._batch_size].to(self._device)
                out[start : start + xb.shape[0]] = (
                    self._model(xb).softmax(dim=1).cpu().numpy()
                )
        return out

    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise RuntimeError("save called before fit.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / _MODEL_FILENAME)
        self._write_meta(
            out,
            extra={
                "in_dim": int(self._in_dim),
                "hidden_dim": self._hidden_dim,
                "num_layers": self._num_layers,
            },
        )

    @classmethod
    def load(cls, path: Path | str) -> "MLPPixelClassifier":
        meta = cls._read_meta(path)
        obj = cls(
            num_classes=int(meta["num_classes"]),
            hidden_dim=int(meta["hidden_dim"]),
            num_layers=int(meta["num_layers"]),
        )
        obj._in_dim = int(meta["in_dim"])
        model = _PointwiseMLP(obj._in_dim, obj._hidden_dim, obj._num_layers, obj._num_classes)
        state = torch.load(Path(path) / _MODEL_FILENAME, map_location=obj._device, weights_only=True)
        model.load_state_dict(state)
        obj._model = model.to(obj._device).eval()
        return obj


pixel_classifier_registry.register("mlp", MLPPixelClassifier)
