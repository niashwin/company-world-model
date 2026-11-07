"""Observations -> world states -> token ids.

World state z = (account, quarter): what is going on with one customer in one quarter. Every observation
(message, email, ticket, call turn, CRM record, issue, meeting) that the company graph links to that account and
falls in that quarter is one *rendering* of z, with its own nuisance (author, system template, wording).

Two system families: A = messages people write to each other (Slack, Gmail),
B = systems of record (Zendesk, Salesforce, Jira, Gong, Calendar).
"""
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

MOD_A = ("slack", "gmail")
MOD_B = ("zendesk", "salesforce", "jira", "gong", "calendar")
TOKEN = re.compile(r"[a-z0-9$][a-z0-9_.$%'-]*[a-z0-9%]|[a-z0-9]")
N_BUCKETS = 1 << 18


def quarter(ts):
    d = datetime.fromtimestamp(ts, timezone.utc)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def tokenize(text):
    return TOKEN.findall((text or "").lower())


@dataclass
class Corpus:
    obs: list                      # observation dicts
    tok: list                      # list[np.ndarray[int32]] unigram ids
    vocab: dict                    # token -> id (< n_uni)
    state: np.ndarray              # state index per obs (-1 if unlinked)
    states: list                   # state index -> (account, quarter)
    account: np.ndarray            # account index per obs (-1 if unlinked)
    accounts: list                 # account index -> account id
    system: np.ndarray             # system index per obs
    systems: list
    modality: np.ndarray           # 0 = A (messages), 1 = B (records)
    author: np.ndarray             # author index per obs
    thread: np.ndarray             # thread index per obs (documents that quote each other)
    ts: np.ndarray
    pools: dict = field(default_factory=dict)   # state -> obs indices; (state, modality) -> obs indices


def load(path, linked_only=True, vocab=None, accounts=None, states=None, max_words=96):
    obs = [json.loads(line) for line in open(path)]
    obs = [o for o in obs if o.get("ts") and o.get("text")]
    if linked_only:
        obs = [o for o in obs if o.get("account_id")]
    toks = [tokenize(o["text"])[:max_words] for o in obs]
    if vocab is None:
        cnt = Counter(t for ts_ in toks for t in ts_)
        vocab = {w: i + 1 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}  # 0 = unknown
    tok = [np.array([vocab.get(t, 0) for t in ts_], dtype=np.int64) for ts_ in toks]
    accounts = list(accounts) if accounts is not None else sorted({o["account_id"] for o in obs if o.get("account_id")})
    a_ix = {a: i for i, a in enumerate(accounts)}
    states = list(states) if states is not None else sorted({(o["account_id"], quarter(o["ts"])) for o in obs if o.get("account_id")})
    s_ix = {s: i for i, s in enumerate(states)}
    systems = sorted({o["system"] for o in obs})
    sys_ix = {s: i for i, s in enumerate(systems)}
    authors = {}
    threads = {}
    st = np.array([s_ix.get((o["account_id"], quarter(o["ts"])), -1) if o.get("account_id") else -1 for o in obs])
    c = Corpus(
        obs=obs, tok=tok, vocab=vocab, state=st, states=states,
        account=np.array([a_ix.get(o.get("account_id"), -1) for o in obs]), accounts=accounts,
        system=np.array([sys_ix[o["system"]] for o in obs]), systems=systems,
        modality=np.array([0 if o["system"] in MOD_A else 1 for o in obs]),
        author=np.array([authors.setdefault(str(o.get("author")), len(authors)) for o in obs]),
        thread=np.array([threads.setdefault(f"{o['system']}:{o.get('thread')}", len(threads)) for o in obs]),
        ts=np.array([o["ts"] for o in obs], dtype=np.float64),
    )
    build_pools(c)
    return c


def build_pools(c):
    pools = defaultdict(list)
    for i, (s, m) in enumerate(zip(c.state, c.modality)):
        if s >= 0:
            pools[int(s)].append(i)
            pools[(int(s), int(m))].append(i)
    c.pools = {k: np.array(v) for k, v in pools.items()}


def features(ids, n_uni):
    """Unigram ids plus hashed bigrams, into one bucket space (the 'pixels' of a text)."""
    ids = np.asarray(ids, dtype=np.int64)
    uni = np.minimum(ids, n_uni - 1)
    if len(ids) > 1:
        bi = n_uni + ((ids[:-1] * 1_000_003 + ids[1:] * 7919) % (N_BUCKETS - n_uni))
        return np.concatenate([uni, bi])
    return uni


def time_code(ts, dim=16, t0=1.58e9, scale=86400 * 91.0):
    """Sinusoidal code of a timestamp in quarters since 2020 (shared by stored observations and queries)."""
    q = (np.asarray(ts, dtype=np.float64) - t0) / scale
    freqs = 2.0 ** np.arange(dim // 2) / 4.0
    ang = q[..., None] * freqs * np.pi / 8
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)
