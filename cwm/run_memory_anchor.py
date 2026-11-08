"""Questions that need the state channel.

The 144 test questions of run_memory2.py, asked without the customer's name: every customer name, employee name and
identifier is masked in the stored records, and each question comes with one record of the customer instead (its
"anchor"), taken from a held-out thread (never seen by the state encoder) and from a different system than the one
that answers the question: a ticket in hand, asking whether there was a sales call this quarter. The reader must find
the customer's records by situation, not by name.

Stores (rows are [content | retrieval key]; questions are [question text | anchor]):
  closed   no store, the question text only
  state    good-view state vectors, retrieved and read
  detail   pretrained (all-MiniLM-L6-v2) vectors, retrieved and read
  concat   both channels side by side, retrieved and read
  routed   retrieve with the state channel, read the detail channel
  link     retrieve the anchor's own customer-quarter by its link (an oracle), read the detail channel
  link_state  the same oracle retrieval, reading the state channel instead
  link_prov   the same oracle retrieval, reading only each record's time and system codes (no content)
  place_state, place_detail, place_concat
           route through states: place the anchor at its nearest customer (centroids of linked training records in
           that channel), retrieve 16 records of that customer in the question's quarter, read the detail channel.
           Stored records from held-out threads are indexed under the customer-quarter the same channel places them in.

CORPUS=masked (default) uses the corpus with customers and employees masked; CORPUS=plain is the unmasked control.
Neighbour stores rank records by the content of the finding channel alone (unit-length keys), among records of the
question's quarter; placement and link stores find records of one customer-quarter. The reader's pre-training uses
only links of training threads. link_prov reads only each found record's time and system codes.
OTHER_SYSTEMS=1 additionally ranks records from the anchor's own system last (an exploratory variant, not in the
paper). STORES selects stores, SEEDS the seeds. Writes results/memory_anchor_<corpus>[_xsys]<SUFFIX>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_pools, load, quarter, time_code, tokenize  # noqa: E402
from jepa import Encoder, train_jepa  # noqa: E402
from memory import ANSWERS, Memory, evaluate, fit_store_map, make_questions, pretrain_memory, train_memory  # noqa: E402
from pretrained import minilm_embed  # noqa: E402
from run_extras import heldout_split  # noqa: E402
from run_prh import D1, OUT  # noqa: E402

CORPUS = os.environ.get("CORPUS", "masked")
EPOCHS = int(os.environ.get("EPOCHS", 12))
PRE = int(os.environ.get("PRESTEPS", 250))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3").split(",")]
STORES = os.environ.get("STORES", "closed,state,detail,concat,routed,place_state,place_detail,place_concat,link,"
                                  "link_state,link_prov").split(",")
SYSTEM_OF = {"tickets": "zendesk", "priority": "zendesk", "call": "gong", "stage": "salesforce"}
K = 16
XSYS = bool(int(os.environ.get("OTHER_SYSTEMS", "0")))
LAMBDA = 10.0   # with unit-norm keys, a same-system record scores about -LAMBDA**2 and ranks below all others
TAG = f"{CORPUS}{'_xsys' if XSYS else ''}"
LOG = open(os.path.join(OUT, f"memory_anchor_{TAG}_log.txt"), "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def l2(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def rows(sem, ts, sysid, n_sys, tdim=16):
    oh = np.eye(n_sys, dtype=np.float32)[sysid] if sysid is not None else np.zeros((len(sem), n_sys), np.float32)
    return np.concatenate([sem, time_code(np.asarray(ts, dtype=float), tdim), oh], 1).astype(np.float32)


def state_encoder(c, te):
    """Good-view encoder trained on this corpus with the held-out threads hidden (cached)."""
    tag = "maskall" if CORPUS == "masked" else "plain"
    pe, px = os.path.join(OUT, f"enc_good_{tag}_ho_s0.pt"), os.path.join(OUT, f"emb_good_{tag}_ho_s0.npy")
    enc = Encoder(len(c.vocab) + 1)
    if os.path.exists(pe) and os.path.exists(px):
        enc.load_state_dict(torch.load(pe))
        return enc, np.load(px)
    true_state = c.state.copy()
    c.state[te] = -1; build_pools(c)
    t = time.time()
    enc, _ = train_jepa(c, "good", steps=1500, seed=0)
    c.state = true_state; build_pools(c)
    X = enc.embed(c.tok)
    torch.save(enc.state_dict(), pe); np.save(px, X)
    log("state encoder trained", f"{time.time() - t:.0f}s")
    return enc, X


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    path = os.path.join(OUT, "masked_employees_observations.jsonl") if CORPUS == "masked" else D1
    c = load(path)
    plain = load(D1) if CORPUS == "masked" else c
    assert [o["src_id"] for o in c.obs] == [o["src_id"] for o in plain.obs] and np.array_equal(c.state, plain.state)
    assert c.systems == plain.systems and np.array_equal(c.system, plain.system)
    gt = json.load(open(D1.replace("observations.jsonl", "ground_truth.json")))
    names = {a["Id"]: a["Name"] for a in gt["accounts"]}
    tr, te = heldout_split(c)
    is_test = np.zeros(len(c.obs), bool); is_test[te] = True
    n_sys = len(c.systems)
    systems = np.array([c.obs[i]["system"] for i in range(len(c.obs))])

    # ---------------- the two channels, for every stored record
    enc, Xs = state_encoder(c, te)
    tok = lambda texts: [np.array([c.vocab.get(w, 0) for w in tokenize(s)][:96], dtype=np.int64) for s in texts]  # noqa: E731
    cache = os.path.join(OUT, "minilm_maskall_day1.npy" if CORPUS == "masked" else "minilm_day1.npy")
    if not os.path.exists(cache):
        np.save(cache, minilm_embed([o["text"] for o in c.obs]))
    Xd = np.load(cache)
    assert len(Xs) == len(Xd) == len(c.obs)
    Xc = np.concatenate([l2(Xs), l2(Xd)], 1)
    emb = {"state": lambda t: enc.embed(tok(t)), "detail": lambda t: minilm_embed(t)}
    emb["concat"] = lambda t: np.concatenate([l2(emb["state"](t)), l2(emb["detail"](t))], 1)
    X = {"state": Xs, "detail": Xd, "concat": Xc}
    fmap = {k: fit_store_map(v) for k, v in X.items()}
    content_rows = {k: rows(fmap[k](X[k]), c.ts, c.system, n_sys) for k in X}
    content_rows["prov"] = rows(np.zeros((len(c.obs), 112), np.float32), c.ts, c.system, n_sys)
    QUARTERS = sorted({s_[1] for s_ in c.states})
    qidx = {q_: i for i, q_ in enumerate(QUARTERS)}
    rec_q = [quarter(t) if quarter(t) in qidx else c.states[c.state[i]][1] for i, t in enumerate(c.ts)]

    # ---------------- questions without names, the same split as run_memory2.py
    qs = make_questions(gt, c.states)
    n_all = {sp: 0 for sp in ("train", "val", "test")}
    rng = np.random.default_rng(0)
    states = sorted({(q["account"], q["quarter"]) for q in qs})
    perm = rng.permutation(len(states))
    split = {states[i]: ("train" if k < 0.7 * len(states) else "val" if k < 0.8 * len(states) else "test")
             for k, i in enumerate(perm)}
    for q in qs:
        q["split"] = split[(q["account"], q["quarter"])]
        q["text"] = q["text"].replace(names[q["account"]], "this customer")
        n_all[q["split"]] += 1
    sid = {s: i for i, s in enumerate(c.states)}
    members = {}
    for i in range(len(c.obs)):
        members.setdefault(int(c.state[i]), []).append(i)
    cands = [[i for i in members.get(sid[(q["account"], q["quarter"])], [])
              if is_test[i] and systems[i] != SYSTEM_OF[q["kind"]]] for q in qs]
    keep = [j for j, q in enumerate(qs) if cands[j]]
    qs = [qs[j] for j in keep]; cands = [cands[j] for j in keep]
    for i, q in enumerate(qs):
        q["qi"] = i
    train_q = [q for q in qs if q["split"] == "train"]
    val_q = [q for q in qs if q["split"] == "val"]
    test_q = [q for q in qs if q["split"] == "test"]
    setup = {"corpus": CORPUS, "questions": len(qs), "train_questions": len(train_q), "val_questions": len(val_q),
             "test_questions": len(test_q), "questions_before_anchoring": n_all,
             "epochs": EPOCHS, "pretrain_steps": PRE, "k": K}
    if os.environ.get("PREP"):
        log("prepared", json.dumps(setup))
        return
    log("setup", json.dumps(setup))
    qtext = [q["text"] for q in qs]
    q_ts = np.array([q["ts"] for q in qs])
    q_content = {k: rows(fmap[k](emb[k](qtext)), q_ts, None, n_sys) for k in X}
    what = {k: rows(fmap[k](emb[k](["What happened with this customer?"])), [0.0], None, n_sys)[0, :-(16 + n_sys)]
            for k in X}

    bank = torch.from_numpy(minilm_embed([a for k in ANSWERS for a in ANSWERS[k]]))
    kind_ix, o = {}, 0
    for k in ANSWERS:
        kind_ix[k] = torch.arange(o, o + len(ANSWERS[k])); o += len(ANSWERS[k])

    # ---------------- placement: each record's nearest customer in each channel (centroids of linked training records)
    train_rec = ~is_test
    placed = {}
    for ch, M in X.items():
        Z = l2(M)
        cen = l2(np.stack([Z[train_rec & (c.account == a)].mean(0) for a in range(len(c.accounts))]))
        placed[ch] = np.argmax(Z @ cen.T, 1)
    quarter_of = {i: s_[1] for i, s_ in enumerate(c.states)}

    def placed_state(ch, i, quarter=None):
        """State index for record i as placed by channel ch (its own link if it is a linked training record)."""
        if quarter is None and train_rec[i]:
            return int(c.state[i])
        q_ = quarter if quarter is not None else quarter_of[int(c.state[i])]
        return sid.get((c.accounts[placed[ch][i]], q_), -1)

    def key_rows(store, idx, zero_system=False, quarter=None):
        """Retrieval keys for records idx: the key channel (state for routed, the store's own channel otherwise).
        With OTHER_SYSTEMS, keys are unit-normalized and get a system block: +LAMBDA for a stored record's system,
        -LAMBDA for the anchor's, so records from the anchor's system rank last and the rest keep their cosine order."""
        idx = np.asarray(idx)
        if store.startswith("link"):
            k = np.zeros((len(idx), len(c.states)), np.float32)
            k[np.arange(len(idx)), c.state[idx]] = 1.0
        elif store.startswith("place_"):
            ch = store[len("place_"):]
            k = np.zeros((len(idx), len(c.states)), np.float32)
            for j, i in enumerate(idx):
                st_ = placed_state(ch, int(i), quarter if zero_system else None)
                if st_ >= 0:
                    k[j, st_] = 1.0
        else:
            ch = "state" if store == "routed" else store
            sem = fmap[ch](X[ch][idx])
            k = np.zeros((len(idx), sem.shape[1] + len(QUARTERS)), np.float32)
            k[:, :sem.shape[1]] = l2(sem)                     # content only, unit length
            qs_ = [quarter if (zero_system and quarter is not None) else rec_q[i] for i in idx]
            k[np.arange(len(idx)), sem.shape[1] + np.array([qidx[q_] for q_ in qs_])] = 10.0   # the quarter first
        if not XSYS:
            return k
        block = np.eye(n_sys, dtype=np.float32)[c.system[idx]] * (-LAMBDA if zero_system else LAMBDA)
        return np.concatenate([l2(k), block], 1).astype(np.float32)

    def content_of(store):
        if store == "link_state":
            return "state"
        if store == "link_prov":
            return "prov"
        return "detail" if store in ("routed", "link", "closed") or store.startswith("place_") else store

    out_path = os.path.join(OUT, f"memory_anchor_{TAG}{os.environ.get('SUFFIX', '')}.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    results["setup"] = setup
    results["heldout_placement_customer"] = {ch: float((placed[ch][te] == c.account[te]).mean()) for ch in X}
    runs = results.setdefault("runs", {})
    retr = results.setdefault("retrieval", {})
    for seed in SEEDS:
        arng = np.random.default_rng(1000 + seed)
        anchor = np.array([int(arng.choice(cd)) for cd in cands])
        for store in STORES:
            t0 = time.time()
            ch = content_of(store)
            S_c = content_rows[ch]
            d = S_c.shape[1]
            qc = q_content["detail" if ch == "prov" else ch]
            if store == "closed":
                S, Q, split_at, variant = torch.from_numpy(S_c), torch.from_numpy(qc), None, "none"
            else:
                S_k = key_rows(store, np.arange(len(c.obs)))
                if store.startswith("link") or store.startswith("place_"):   # break ties at random
                    noise = 0.01 * np.random.default_rng(seed).random((len(S_k), len(c.states))).astype(np.float32)
                    S_k[:, :len(c.states)] += noise
                if store.startswith("link"):
                    Q_k = key_rows(store, anchor, zero_system=True)
                else:    # the anchor gives the customer, the question gives the quarter
                    Q_k = np.concatenate([key_rows(store, [a_], zero_system=True, quarter=q["quarter"])
                                          for a_, q in zip(anchor, qs)])
                S = torch.from_numpy(np.concatenate([S_c, S_k], 1))
                Q = torch.from_numpy(np.concatenate([qc, Q_k], 1))
                split_at, variant = d, "topk"
                # which records each test question retrieves
                te_ix = np.array([q["qi"] for q in test_q])
                sim = torch.nn.functional.normalize(Q[te_ix, d:], dim=-1) @ torch.nn.functional.normalize(S[:, d:], dim=-1).T
                top = sim.topk(K, dim=1).indices.numpy()
                same_c = [np.mean(c.account[top[j]] == c.account[anchor[qi]]) for j, qi in enumerate(te_ix)]
                same_s = [np.mean(c.state[top[j]] == c.state[anchor[qi]]) for j, qi in enumerate(te_ix)]
                useful = [np.mean((c.state[top[j]] == c.state[anchor[qi]]) & (systems[top[j]] == SYSTEM_OF[qs[qi]["kind"]]))
                          for j, qi in enumerate(te_ix)]
                same_t = [np.mean(c.thread[top[j]] == c.thread[anchor[qi]]) for j, qi in enumerate(te_ix)]
                other_sys = [np.mean(systems[top[j]] != systems[anchor[qi]]) for j, qi in enumerate(te_ix)]
                retr[f"{store}/s{seed}"] = {"same_customer": float(np.mean(same_c)), "same_state": float(np.mean(same_s)),
                                            "same_state_answering_system": float(np.mean(useful)),
                                            "anchor_thread": float(np.mean(same_t)), "other_systems": float(np.mean(other_sys))}
            torch.manual_seed(seed)
            mem = Memory(d, bank.size(1), variant=variant, split=split_at)
            if store != "closed":
                sem = S_c[:, :-(16 + n_sys)]

                def make_query(a, t, r, store=store, ch=ch):
                    w = what["detail" if ch == "prov" else ch]
                    content = np.concatenate([w, time_code(np.array([t]), 16)[0], np.zeros(n_sys, np.float32)])
                    key = key_rows(store, [r], zero_system=True, quarter=rec_q[r])[0]
                    return np.concatenate([content, key]).astype(np.float32)
                acct_known = c.account.copy(); acct_known[is_test] = -1      # pre-training sees training links only
                pretrain_memory(mem, S, sem, c.ts, acct_known, c.system, n_sys, None, steps=PRE, seed=seed, log=log,
                                make_query=make_query)
            mem = train_memory(S, Q, train_q, bank, kind_ix, variant, epochs=EPOCHS, batch=32, seed=seed, log=log,
                               val=val_q, init=mem)
            r = evaluate(mem, S, Q, test_q, bank, kind_ix)
            r["seconds"] = time.time() - t0
            runs[f"{store}/s{seed}"] = r
            log(store, seed, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}),
                json.dumps(retr.get(f"{store}/s{seed}", {})))
            json.dump(results, open(out_path, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
