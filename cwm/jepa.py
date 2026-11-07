"""Text JEPAs trained from scratch on one company's observations, with good views, bad views or next-token prediction.

Follows Balestriero & LeCun (arXiv 2610.39302) and LeJEPA (2511.08544):
  * loss = (1 - lam) * invariance to the mean of the global views + lam * SIGReg (sketched isotropic Gaussian)
  * views are trimmed (global 50-100%, local 15-50% contiguous spans) and lightly masked (15%), 2 global + 6 local
  * GOOD views: every view draws its OWN observation of the same world state (account x quarter)
  * BAD views:  every view corrupts the SAME observation
  * NTP:        a prefix predicts its next token (softmax head) - a JEPA whose positives are prefix/continuation
The encoder is a bag of unigrams + hashed bigrams with an MLP: cheap enough to train on a laptop CPU, and it has to
learn from the company's data alone (no pretrained language knowledge leaks in).
"""
import math
import time

import numpy as np
import torch
from torch import nn

from data import N_BUCKETS, features

torch.set_num_threads(max(1, torch.get_num_threads()))


class Encoder(nn.Module):
    def __init__(self, n_uni, d_emb=192, d=256):
        super().__init__()
        self.n_uni = n_uni
        self.bag = nn.EmbeddingBag(N_BUCKETS, d_emb, mode="mean", sparse=True)
        nn.init.normal_(self.bag.weight, std=0.1)
        self.net = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, 512), nn.GELU(), nn.Linear(512, d))
        self.d = d

    def forward(self, flat, offsets):
        return self.net(self.bag(flat, offsets))

    @torch.no_grad()
    def embed(self, toks, batch=4096):
        self.eval()
        out = []
        for i in range(0, len(toks), batch):
            flat, off = pack([features(t, self.n_uni) if len(t) else np.array([0]) for t in toks[i:i + batch]])
            out.append(self(flat, off))
        self.train()
        return torch.cat(out).numpy()


def pack(feats):
    lens = np.array([max(1, len(f)) for f in feats])
    flat = np.concatenate([f if len(f) else np.array([0]) for f in feats])
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return torch.from_numpy(flat.astype(np.int64)), torch.from_numpy(off.astype(np.int64))


def masked_features(ids, n_uni):
    """Features of a span whose masked tokens are -1: unigrams and bigrams that avoid the masks."""
    keep = ids >= 0
    uni = np.minimum(ids[keep], n_uni - 1)
    if len(ids) > 1:
        ok = keep[:-1] & keep[1:]
        a, b = ids[:-1][ok], ids[1:][ok]
        bi = n_uni + ((a * 1_000_003 + b * 7919) % (N_BUCKETS - n_uni))
        f = np.concatenate([uni, bi])
    else:
        f = uni
    return f if len(f) else np.array([0])


def text_view(ids, scale, mask, rng):
    n = len(ids)
    if n == 0:
        return np.array([0])
    keep = max(1, int(round(n * rng.uniform(*scale))))
    start = rng.integers(0, n - keep + 1)
    span = ids[start:start + keep].copy()
    span[rng.random(keep) < mask] = -1
    return span


class Projector(nn.Module):
    def __init__(self, d=256, h=1024, k=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


def sigreg(z, step, num_slices=256, knots=17):
    """Sketched Epps-Pulley test of z against N(0, I), averaged over random 1-D slices (LeJEPA)."""
    g = torch.Generator().manual_seed(int(step))
    A = torch.randn(z.size(1), num_slices, generator=g)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(-5, 5, knots)
    phi = torch.exp(-0.5 * t ** 2)
    xt = (z @ A).unsqueeze(-1) * t
    err = (torch.cos(xt).mean(0) - phi) ** 2 + torch.sin(xt).mean(0) ** 2
    return (torch.trapezoid(err * phi, t, dim=-1) * z.size(0)).mean()


def train_jepa(c, views="good", modality=None, steps=1500, batch=256, n_global=2, n_local=6, lam=0.05, lr=1e-3,
               gscale=(0.5, 1.0), lscale=(0.15, 0.5), mask=0.15, seed=0, d=256, log=None, obs_subset=None):
    """views: 'good' | 'bad'. modality: None (all systems) or 0/1 (train only on that modality)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_uni = len(c.vocab) + 1
    enc, proj = Encoder(n_uni, d=d), Projector(d)
    opt_s = torch.optim.SparseAdam(list(enc.bag.parameters()), lr=lr)
    dense = list(enc.net.parameters()) + list(proj.parameters())
    opt_d = torch.optim.AdamW(dense, lr=lr, weight_decay=0.05)
    linked = np.where(c.state >= 0)[0]
    if modality is not None:
        linked = linked[c.modality[linked] == modality]
    if obs_subset is not None:
        linked = np.intersect1d(linked, obs_subset)
    t0 = time.time()
    hist = []
    for step in range(steps):
        frac = step / steps
        cur = lr * min(1.0, (step + 1) / (0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * frac))
        for o in (opt_s, opt_d):
            for gopt in o.param_groups:
                gopt["lr"] = cur
        anchors = rng.choice(linked, batch)
        feats = []
        for v in range(n_global + n_local):
            scale = gscale if v < n_global else lscale
            for a in anchors:
                if views == "good":
                    key = int(c.state[a]) if modality is None else (int(c.state[a]), modality)
                    pool = c.pools[key]
                    src = pool[rng.integers(len(pool))]
                else:
                    src = a
                feats.append(masked_features(text_view(c.tok[src], scale, mask, rng), n_uni))
        flat, off = pack(feats)
        z = proj(enc(flat, off)).view(n_global + n_local, batch, -1)
        centers = z[:n_global].mean(0)
        inv = (z - centers).square().mean()
        sig = torch.stack([sigreg(z[v], step * 16 + v) for v in range(z.size(0))]).mean()
        loss = (1 - lam) * inv + lam * sig
        opt_s.zero_grad(); opt_d.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        opt_s.step(); opt_d.step()
        if step % 250 == 0 or step == steps - 1:
            hist.append((step, float(inv.detach()), float(sig.detach())))
            if log:
                log(f"  [{views} m={modality}] step {step} inv {float(inv):.4f} sigreg {float(sig):.3f} ({time.time()-t0:.0f}s)")
    return enc, hist


def train_ntp(c, modality=None, steps=1500, batch=1024, vocab_out=8192, lr=1e-3, seed=0, d=256, log=None, obs_subset=None):
    """Next-token prediction with the same encoder: a prefix (bag) predicts the next token (softmax head)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_uni = len(c.vocab) + 1
    enc = Encoder(n_uni, d=d)
    head = nn.Linear(d, vocab_out)
    opt_s = torch.optim.SparseAdam(list(enc.bag.parameters()), lr=lr)
    dense = list(enc.net.parameters()) + list(head.parameters())
    opt_d = torch.optim.AdamW(dense, lr=lr, weight_decay=0.05)
    linked = np.where(c.state >= 0)[0]
    if modality is not None:
        linked = linked[c.modality[linked] == modality]
    if obs_subset is not None:
        linked = np.intersect1d(linked, obs_subset)
    linked = np.array([i for i in linked if len(c.tok[i]) >= 2])
    t0 = time.time()
    for step in range(steps):
        frac = step / steps
        cur = lr * min(1.0, (step + 1) / (0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * frac))
        for o in (opt_s, opt_d):
            for gopt in o.param_groups:
                gopt["lr"] = cur
        docs = rng.choice(linked, batch)
        feats, tgt = [], []
        for i in docs:
            ids = c.tok[i]
            cut = int(rng.integers(1, len(ids)))
            # targets outside the output vocabulary are re-drawn once, else mapped to the 'rare' class 0
            y = int(ids[cut])
            feats.append(features(ids[:cut], n_uni))
            tgt.append(y if y < vocab_out else 0)
        flat, off = pack(feats)
        logits = head(enc(flat, off))
        loss = nn.functional.cross_entropy(logits, torch.tensor(tgt))
        opt_s.zero_grad(); opt_d.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        opt_s.step(); opt_d.step()
        if log and (step % 250 == 0 or step == steps - 1):
            log(f"  [ntp m={modality}] step {step} ce {float(loss):.3f} ({time.time()-t0:.0f}s)")
    return enc, []
