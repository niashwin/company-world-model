"""Memorilla-style latent memory over a company's observation store (Fallahpour et al., arXiv 2610.21375).

Store: every observation encoded ONCE by a frozen encoder (plus a time code), never re-encoded as the store grows.
Read:  K learned slots -> conditioned on the query by cross-attention -> L cross-attention blocks over the WHOLE store
       (softmax over documents, Perceiver-style) -> LayerNorm -> projection to the consumer's input space.
Consumer: frozen. Here (CPU prototype) it is the frozen answer-embedding space of a pretrained sentence encoder,
       scored like PersonaMem-v2 short-answer accuracy (the nearest predefined answer wins); in production it is a
       frozen open-weight LLM reading the K memory tokens before the question (paper, Sec. 2.3).
Variants (paper, App. E.1): no memory (closed book), global mean-pool, cosine top-K, MEMORILLA.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import time_code


# ----------------------------------------------------------------------------------------- questions from ground truth
def q_label(qs):
    return f"Q{qs[-1]} {qs[:4]}"


def q_bounds(qs):
    from datetime import datetime, timezone
    y, k = int(qs[:4]), int(qs[-1])
    start = datetime(y, 3 * (k - 1) + 1, 1, tzinfo=timezone.utc).timestamp()
    end = datetime(y + (k == 4), (3 * k) % 12 + 1, 1, tzinfo=timezone.utc).timestamp()
    return start, end


ANSWERS = {
    "tickets": ["none", "one or two", "three to five", "six or more"],
    "call": ["yes", "no"],
    "stage": ["no open deal", "prospecting", "qualification", "proposal", "negotiation", "closed won", "closed lost"],
    "priority": ["no tickets", "low", "normal", "high", "urgent"],
}
TEMPLATES = {
    "tickets": "How many support tickets did {name} open in {q}?",
    "call": "Did we have a call with {name} in {q}?",
    "stage": "What stage was the {name} deal in at the end of {q}?",
    "priority": "What was the most severe ticket priority for {name} in {q}?",
}


def _ts(s):
    from era_export import iso_ts
    return iso_ts(s)


def make_questions(gt, states):
    names = {a["Id"]: a["Name"] for a in gt["accounts"]}
    org2acct = {o["id"]: o["external_id"] for o in gt["zendesk_orgs"]}
    tickets = [(org2acct.get(t["organization_id"]), _ts(t["created_at"]), (t.get("priority") or "normal").lower())
               for t in gt["tickets"]]
    calls = [(c["account_id"], _ts(c["started"])) for c in gt["calls"]]
    hist = {}
    for h in gt["opportunity_history"]:
        hist.setdefault(h["OpportunityId"], []).append((_ts(h["CreatedDate"]), h["StageName"]))
    opps = {}
    for o in gt["opportunities"]:
        opps.setdefault(o["AccountId"], []).append((_ts(o["CreatedDate"]), o["Id"], o["StageName"]))
    rank = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
    out = []
    for (acct, qs) in states:
        if acct not in names:
            continue
        lo, hi = q_bounds(qs)
        tk = [p for a, t, p in tickets if a == acct and t is not None and lo <= t < hi]
        n = len(tk)
        ans = {"tickets": "none" if n == 0 else "one or two" if n <= 2 else "three to five" if n <= 5 else "six or more",
               "call": "yes" if any(a == acct and t is not None and lo <= t < hi for a, t in calls) else "no",
               "priority": "no tickets" if n == 0 else max(tk, key=lambda p: rank.get(p, 1))}
        stage = "no open deal"
        live = sorted([(c, i) for c, i, _ in opps.get(acct, []) if c is not None and c < hi])
        if live:
            _, oid = live[-1]
            ev = sorted([(t, s) for t, s in hist.get(oid, []) if t is not None and t < hi])
            if ev:
                stage = ev[-1][1].lower()
        ans["stage"] = stage if stage in ANSWERS["stage"] else "no open deal"
        for kind, a in ans.items():
            if a not in ANSWERS[kind]:
                continue
            out.append(dict(kind=kind, account=acct, quarter=qs, answer=a, answer_ix=ANSWERS[kind].index(a),
                            text=TEMPLATES[kind].format(name=names[acct], q=q_label(qs)), ts=(lo + hi) / 2))
    return out


# ----------------------------------------------------------------------------------------- memory modules
class StoreAttention(nn.Module):
    """Cross-attention from B x K slots to ONE shared store (N x h): keys/values are computed once per step,
    softmax over documents, so the store is never copied per query."""

    def __init__(self, h, heads, dropout):
        super().__init__()
        self.H, self.dh = heads, h // heads
        self.q, self.k, self.v, self.o = (nn.Linear(h, h) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, R, Z, return_attn=False):
        B, K, h = R.shape
        q = self.q(R).view(B, K, self.H, self.dh).transpose(1, 2)            # B H K dh
        if Z.dim() == 2:                                                     # shared store
            k = self.k(Z).view(-1, self.H, self.dh).transpose(0, 1)          # H N dh
            v = self.v(Z).view(-1, self.H, self.dh).transpose(0, 1)
            att = torch.einsum("bhkd,hnd->bhkn", q, k) / math.sqrt(self.dh)
            att = att.softmax(-1)
            out = torch.einsum("bhkn,hnd->bhkd", att, v)          # no dropout on a 22K-wide attention map
        else:                                                                # per-query candidates (top-k)
            k = self.k(Z).view(B, -1, self.H, self.dh).transpose(1, 2)       # B H n dh
            v = self.v(Z).view(B, -1, self.H, self.dh).transpose(1, 2)
            att = (q @ k.transpose(-1, -2) / math.sqrt(self.dh)).softmax(-1)
            out = self.drop(att) @ v
        out = self.o(out.transpose(1, 2).reshape(B, K, h))
        return (out, att) if return_attn else out


class Memory(nn.Module):
    def __init__(self, d_in, d_out, variant="memorilla", K=16, h=64, heads=4, layers=2, dropout=0.1, topk=16, split=None):
        super().__init__()
        self.variant, self.K, self.topk = variant, K, topk
        self.split = split          # routing: rows are [content | key]; retrieve by the key, read the content
        self.doc_in = nn.Identity() if d_in == h else nn.Linear(d_in, h, bias=False)
        self.q_in = nn.Linear(d_in, h)
        self.slots = nn.Parameter(torch.empty(K, h))
        nn.init.xavier_uniform_(self.slots)
        self.q_attn = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.LayerNorm(h)
        self.doc_norm = nn.LayerNorm(h)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(h), attn=StoreAttention(h, heads, dropout),
                n2=nn.LayerNorm(h), ffn=nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Dropout(dropout),
                                                      nn.Linear(4 * h, h), nn.Dropout(dropout)))))
        self.out_norm = nn.LayerNorm(h)
        self.proj = nn.Linear(h, d_out)

    def forward(self, store, q, store_raw=None, q_raw=None, return_attn=False, return_state=False):
        if self.split is not None:
            store, store_raw = store[:, :self.split], store[:, self.split:]
            q, q_raw = q[:, :self.split], q[:, self.split:]
        B = q.size(0)
        zq = self.q_in(q).unsqueeze(1)                                     # B x 1 x h
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)
        R = self.q_norm(slots + self.q_attn(slots, zq, zq, need_weights=False)[0])   # query-conditioned slots
        attn = []
        if self.variant != "none":                                         # 'none' = closed book
            Z = self.doc_norm(self.doc_in(store))                          # N x h  (whole store)
            if self.variant == "mean":
                R = R + Z.mean(0).view(1, 1, -1)                           # one collection vector for every slot
                Zb = None
            elif self.variant == "topk":                                   # hard cosine top-k, no learning
                sim = F.normalize(q_raw, dim=-1) @ F.normalize(store_raw, dim=-1).T
                Zb = Z[sim.topk(self.topk, dim=1).indices]                 # B x k x h
            else:
                Zb = Z                                                     # every slot reads every document
            for blk in self.blocks:
                if Zb is not None:
                    o = blk["attn"](blk["n1"](R), Zb, return_attn=return_attn)
                    if return_attn:
                        o, a = o
                        attn.append(a)
                    R = R + o
                R = R + blk["ffn"](blk["n2"](R))
        Rn = self.out_norm(R)
        M = self.proj(Rn)                                                  # B x K x d_out (soft memory tokens)
        if return_state:
            return M, Rn
        return (M, attn) if return_attn else M


def consumer_logits(M, answer_bank, temperature=0.07):
    """Frozen consumer: the K memory tokens are read as one vector and matched to the predefined answers."""
    v = F.normalize(M.mean(1), dim=-1)
    return v @ answer_bank.T / temperature


def fit_store_map(X, k=112):
    """Frozen map from an encoder's space to the store: PCA-whitening to k dims, fitted once on the store."""
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:k].T / (S[:k] / np.sqrt(len(X)) + 1e-6)
    return lambda Y: ((Y - mu) @ W / np.sqrt(k)).astype(np.float32)


def to_inputs(Xmapped, ts, tdim=16):
    return torch.from_numpy(np.concatenate([Xmapped, time_code(ts, tdim)], 1).astype(np.float32))


def train_memory(store, Q, qs_train, answer_bank, kind_ix, variant, epochs=40, batch=16, lr=1e-3, seed=0, log=None,
                 val=None, init=None):
    """kind_ix: dict kind -> LongTensor of answer-bank rows for that question type."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mem = init if init is not None else Memory(store.size(1), answer_bank.size(1), variant=variant)
    opt = torch.optim.AdamW(mem.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(qs_train) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state = -1, None
    for ep in range(epochs):
        perm = rng.permutation(len(qs_train))
        for i in range(0, len(perm), batch):
            b = [qs_train[j] for j in perm[i:i + batch]]
            qi = torch.tensor([x["qi"] for x in b])
            M = mem(store, Q[qi], store_raw=store, q_raw=Q[qi])
            logits = consumer_logits(M, answer_bank)
            loss = 0
            for j, x in enumerate(b):
                rows = kind_ix[x["kind"]]
                loss = loss + F.cross_entropy(logits[j, rows].unsqueeze(0), torch.tensor([x["answer_ix"]]))
            loss = loss / len(b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
            opt.step(); sched.step()
        if val is not None and (ep % 3 == 2 or ep == epochs - 1):
            acc = evaluate(mem, store, Q, val, answer_bank, kind_ix)["overall"]
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in mem.state_dict().items()}
            if log:
                log(f"    [{variant}] epoch {ep+1} loss {float(loss):.3f} val {acc:.3f}")
    if best_state is not None:
        mem.load_state_dict(best_state)
    return mem


@torch.no_grad()
def evaluate(mem, store, Q, qs, answer_bank, kind_ix, batch=32):
    mem.eval()
    correct = {}
    for i in range(0, len(qs), batch):
        b = qs[i:i + batch]
        qi = torch.tensor([x["qi"] for x in b])
        logits = consumer_logits(mem(store, Q[qi], store_raw=store, q_raw=Q[qi]), answer_bank)
        for j, x in enumerate(b):
            pred = int(logits[j, kind_ix[x["kind"]]].argmax())
            correct.setdefault(x["kind"], []).append(pred == x["answer_ix"])
    mem.train()
    out = {k: float(np.mean(v)) for k, v in correct.items()}
    out["overall"] = float(np.mean([v for vs in correct.values() for v in vs]))
    return out


# ----------------------------------------------------------------------------------------- stage-1 style pretraining
def recall_targets(sem, ts, acct, sysid, n_sys, a, t, sigma=30 * 86400.0):
    """What the store holds about account `a` around time `t`: a time-weighted mean of its records' state vectors,
    the share of each source system, and log volume per system. Built from the store itself - no answers."""
    idx = np.where(acct == a)[0]
    w = np.exp(-0.5 * ((ts[idx] - t) / sigma) ** 2)
    tot = w.sum()
    if tot < 1e-3:
        return None
    m = (w[:, None] * sem[idx]).sum(0) / tot
    vol = np.zeros(n_sys)
    np.add.at(vol, sysid[idx], w)
    return np.concatenate([m / (np.linalg.norm(m) + 1e-9), vol / tot, np.log1p(vol)]).astype(np.float32)


def pretrain_memory(mem, store, sem, ts, acct, sysid, n_sys, account_queries, steps=400, batch=32, lr=1e-3, seed=0,
                    log=None, tdim=16, make_query=None):
    """Memorilla's stage 1 adapted to a company: from a question naming a customer and a date, recover what the
    store holds about that customer around that date. Unlimited queries, no labels beyond the link graph."""
    rng = np.random.default_rng(seed)
    d_sem = sem.shape[1]
    head = nn.Linear(mem.out_norm.normalized_shape[0], d_sem + 2 * n_sys)
    params = list(mem.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    linked = np.where(acct >= 0)[0]
    for step in range(steps):
        qs, tg = [], []
        while len(qs) < batch:
            r = rng.choice(linked)
            a, t = int(acct[r]), float(ts[r] + rng.uniform(-30, 30) * 86400)
            y = recall_targets(sem, ts, acct, sysid, n_sys, a, t)
            if y is None:
                continue
            if make_query is not None:
                qs.append(make_query(a, t))
            else:
                qs.append(np.concatenate([account_queries[a], time_code(np.array([t]), tdim)[0], np.zeros(n_sys, np.float32)]))
            tg.append(y)
        q = torch.from_numpy(np.stack(qs))
        y = torch.from_numpy(np.stack(tg))
        _, R = mem(store, q, store_raw=store, q_raw=q, return_state=True)
        out = head(R.mean(1))
        s_pred, s_true = F.normalize(out[:, :d_sem], dim=-1), y[:, :d_sem]
        logits = s_pred @ s_true.T / 0.1                                  # find THIS customer's state, not another's
        loss = (F.cross_entropy(logits, torch.arange(batch))
                + F.kl_div(F.log_softmax(out[:, d_sem:d_sem + n_sys], -1), y[:, d_sem:d_sem + n_sys], reduction="batchmean")
                + F.mse_loss(out[:, d_sem + n_sys:], y[:, d_sem + n_sys:]))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log and (step % 50 == 0 or step == steps - 1):
            acc = float((logits.argmax(1) == torch.arange(batch)).float().mean())
            log(f"    [pretrain] step {step} loss {float(loss):.3f} in-batch recall@1 {acc:.2f}")
    return mem
