"""Fast checks that run without Era data: a tiny synthetic company, a few training steps, the metrics, the access
policy, non-interference of an encoder trained on one reader's records, the good-view head on frozen features, the
fact that a rotation cannot change a reader's placement accuracy, and retrieval by key in the memory reader."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cwm"))

from acl import EVERYONE, members, record_readers  # noqa: E402
from data import build_pools, load  # noqa: E402
from jepa import train_jepa  # noqa: E402
from metrics import knn_precision, procrustes  # noqa: E402

TOPICS = {"acct-0": "invoice export mismatch billing portal", "acct-1": "sso login token refresh outage",
          "acct-2": "renewal pricing discount negotiation"}
SYSTEMS = ["gmail", "zendesk", "salesforce"]
QUARTER_TS = [1.735e9, 1.743e9, 1.751e9]          # three different calendar quarters (2025)


def write_company(path, secret=None):
    rng = np.random.default_rng(0)
    rows = []
    for a, topic in TOPICS.items():
        for q, ts in enumerate(QUARTER_TS):
            for s in SYSTEMS:
                for j in range(6):
                    words = topic.split()
                    rng.shuffle(words)
                    text = f"update {j} on {' '.join(words)} for quarter {q}"
                    if secret and s == "salesforce" and a == "acct-1":
                        text += f" {secret}"
                    rows.append(dict(system=s, kind="x", ts=ts + 3600 * (j + 7 * SYSTEMS.index(s)),
                                     author="jodid@northwind.com" if s == "gmail" else "x", text=text,
                                     account_id=a, thread=f"{a}-{q}-{s}-{j // 2}", channel=None,
                                     src_id=f"{a}-{q}-{s}-{j}", meta={}))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def test_train_and_metrics(tmp_path):
    p = tmp_path / "obs.jsonl"
    write_company(p)
    c = load(str(p))
    assert len(c.states) == 9 and len(c.accounts) == 3
    enc, hist = train_jepa(c, "good", steps=40, batch=32, seed=0)
    X = enc.embed(c.tok)
    assert X.shape == (len(c.obs), 256) and np.isfinite(X).all()
    q = np.arange(len(c.obs))
    prec = knn_precision(X, c.account, c.system, c.thread, q, k=5)
    assert 0.0 <= prec <= 1.0


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(200, 16))
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    R = procrustes(A, A @ Q.T)          # R minimises ||A - (A Q^T) R||
    assert np.allclose(A @ Q.T @ R, A, atol=1e-8)


def test_access_policy():
    emails = {"jodid@northwind.com": "user-3"}
    assert record_readers({"system": "slack", "channel": "general"}, emails) == EVERYONE
    assert record_readers({"system": "slack", "channel": "D-user-0-user-3"}, emails) == {"user-0", "user-3"}
    assert record_readers({"system": "slack", "channel": "leadership"}, emails) == members({"leadership"})
    crm = record_readers({"system": "salesforce"}, emails)
    assert "user-3" not in crm and "user-9" in crm                  # support cannot read the CRM; sales can
    mail = record_readers({"system": "gmail", "author": "Jodi <jodid@northwind.com>"}, emails)
    assert "user-3" in mail


def _reader_view(path, steps=20):
    """Train on the records a support engineer may read; return its embeddings of those records."""
    c = load(str(path))
    emails = {"jodid@northwind.com": "user-3"}
    vis = [i for i, o in enumerate(c.obs) if "user-3" in record_readers(o, emails)]
    sub = path.parent / (path.stem + "_visible.jsonl")
    with open(sub, "w") as f:
        for i in vis:
            f.write(json.dumps(c.obs[i], sort_keys=True) + "\n")
    cs = load(str(sub))
    build_pools(cs)
    enc, _ = train_jepa(cs, "good", steps=steps, batch=16, seed=0)
    return enc.embed(cs.tok)


@pytest.mark.parametrize("secret", ["codename zephyrine churn risk"])
def test_non_interference(tmp_path, secret):
    a, b = tmp_path / "clean.jsonl", tmp_path / "secret.jsonl"
    write_company(a)
    write_company(b, secret=secret)          # the secret lives only in CRM records support cannot read
    Xa, Xb = _reader_view(a), _reader_view(b)
    assert np.array_equal(Xa, Xb)


def test_pretrained_head_trains(tmp_path):
    """The good-view head trains on frozen backbone features (no download needed: features are random here)."""
    from pretrained import train_head
    p = tmp_path / "obs.jsonl"
    write_company(p)
    c = load(str(p))
    build_pools(c)
    F = np.random.default_rng(0).normal(size=(len(c.obs), 384)).astype(np.float32)
    h = train_head(F, c, steps=5, batch=16, seed=0)
    Z = h.embed(F)
    assert Z.shape == (len(c.obs), 256) and np.isfinite(Z).all()


def test_rotation_does_not_change_placement(tmp_path):
    """Placement at the nearest state centroid (cosine) is the same after any rotation of the reader's space, so a
    rotation can make two readers' views comparable but cannot make either more accurate."""
    from run_acl import placement
    p = tmp_path / "obs.jsonl"
    write_company(p)
    c = load(str(p))
    rng = np.random.default_rng(3)
    X = rng.normal(size=(len(c.obs), 32))
    is_test = rng.random(len(c.obs)) < 0.3
    rows = np.arange(len(c.obs))
    Q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    a, b = placement(X, rows, c, is_test), placement(X @ Q, rows, c, is_test)
    assert a == b


def test_memory_retrieves_by_key():
    """With rows split into [content | key], the reader retrieves by key: the same question with different keys reads
    different records, and with the same key reads the same ones.

    Each key has exactly `topk` records, so the top-k set does not depend on how the CPU breaks ties between equal
    scores. The reader ignores the order of its candidates, so outputs agree up to the order of float sums."""
    import torch
    from memory import Memory
    torch.manual_seed(0)
    n, d, per_key = 40, 8, 20
    keys = torch.zeros(n, 2); keys[:per_key, 0] = 1.0; keys[per_key:, 1] = 1.0
    S = torch.cat([torch.randn(n, d), keys], 1)
    q = torch.randn(1, d).repeat(2, 1)
    mem = Memory(d, 6, variant="topk", split=d, topk=per_key).eval()
    with torch.no_grad():
        out = mem(S, torch.cat([q, torch.tensor([[1.0, 0.0], [0.0, 1.0]])], 1))
        same = mem(S, torch.cat([q, torch.tensor([[1.0, 0.0], [1.0, 0.0]])], 1))
        shuffled = mem(S[torch.randperm(n)], torch.cat([q, torch.tensor([[1.0, 0.0], [0.0, 1.0]])], 1))
    assert out.shape == (2, 16, 6)
    assert not torch.allclose(out[0], out[1], atol=1e-3)
    assert torch.allclose(same[0], same[1], atol=1e-5)
    assert torch.allclose(out, shuffled, atol=1e-5)   # the order of the store does not matter
