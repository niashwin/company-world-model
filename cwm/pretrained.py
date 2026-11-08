"""A pretrained, frozen text backbone (all-MiniLM-L6-v2) with a small JEPA head trained on good views.

The backbone was trained outside the company, so it carries no company data and can be shared by every reader
without breaking access control; only the head is trained per access scope. With precomputed backbone features, a
view is a whole record (its frozen embedding) drawn from the anchor's state, with feature dropout and a little noise
as augmentation. The loss is LeJEPA's, as for the from-scratch encoder.
"""
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jepa import Projector, sigreg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def minilm_embed(texts, batch=128):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, cache_folder=os.path.join(ROOT, "models"), device="cpu")
    return m.encode(list(texts), batch_size=batch, show_progress_bar=False, convert_to_numpy=True,
                    normalize_embeddings=True).astype(np.float32)


class Head(nn.Module):
    def __init__(self, d_in=384, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d))

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def embed(self, feats, batch=8192):
        self.eval()
        out = [self(torch.from_numpy(feats[i:i + batch])) for i in range(0, len(feats), batch)]
        self.train()
        return torch.cat(out).numpy()


def train_head(feats, c, steps=800, batch=256, n_global=2, n_local=6, lam=0.05, lr=1e-3, drop=0.1, noise=0.02,
               seed=0, init=None):
    """Train a JEPA head on frozen features with good views: every view is a record of the anchor's state.
    `feats` is aligned with c.obs; records with state -1 (unlinked or hidden) are never used."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    head = Head(feats.shape[1]) if init is None else init
    proj = Projector(256)
    opt = torch.optim.AdamW(list(head.parameters()) + list(proj.parameters()), lr=lr, weight_decay=0.05)
    X = torch.from_numpy(feats)
    linked = np.where(c.state >= 0)[0]
    V = n_global + n_local
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / (0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = cur
        anchors = rng.choice(linked, batch)
        idx = np.empty((V, batch), dtype=np.int64)
        for j, a in enumerate(anchors):
            pool = c.pools[int(c.state[a])]
            idx[:, j] = pool[rng.integers(len(pool), size=V)]
        x = X[torch.from_numpy(idx.reshape(-1))]
        x = F.dropout(x, drop, training=True) + noise * torch.randn_like(x)
        z = proj(head(x)).view(V, batch, -1)
        centers = z[:n_global].mean(0)
        inv = (z - centers).square().mean()
        sig = torch.stack([sigreg(z[v], step * 16 + v) for v in range(V)]).mean()
        loss = (1 - lam) * inv + lam * sig
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head
