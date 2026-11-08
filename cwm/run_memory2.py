"""E4b: the same Memorilla read, with the two things the first run lacked.

1. Stage-1 style pretraining of the read (Memorilla trains its module on ~3.3M reconstruction and QA examples
   before any task): "recover what the store holds about <customer> around <date>", generated from the store.
2. Source type kept as explicit metadata on each stored row: the good-view encoder deliberately discards which
   system wrote a record (nuisance for state), but "was there a call?" needs it.
Store rows = [state vector (good-view JEPA, PCA-112) ; time code (16) ; source one-hot (7)].
STORE=minilm uses a pretrained sentence encoder instead; STORE=dual puts both side by side (two channels: state and
detail) before the same PCA-112 map; STORE=route retrieves with the state channel and reads the detail channel.
Writes results/memory2.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import load, quarter, time_code, tokenize  # noqa: E402
from jepa import Encoder  # noqa: E402
from memory import (ANSWERS, Memory, evaluate, fit_store_map, make_questions, pretrain_memory,  # noqa: E402
                    train_memory)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
D1, D2 = (os.path.join(ROOT, f"data/era/{d}/observations.jsonl") for d in ("day1", "day2"))
EPOCHS = int(os.environ.get("EPOCHS", 12))
PRE = int(os.environ.get("PRESTEPS", 250))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
SUFFIX = os.environ.get("SUFFIX", "")
LOG = open(os.path.join(OUT, f"memory2{SUFFIX}_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def rows(sem, ts, sysid, n_sys, tdim=16):
    oh = np.eye(n_sys, dtype=np.float32)[sysid] if sysid is not None else np.zeros((len(sem), n_sys), np.float32)
    return torch.from_numpy(np.concatenate([sem, time_code(ts, tdim), oh], 1).astype(np.float32))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    results = {"setup": {"epochs": EPOCHS, "pretrain_steps": PRE}}
    c1 = load(D1)
    gt1 = json.load(open(os.path.join(ROOT, "data/era/day1/ground_truth.json")))
    gt2 = json.load(open(os.path.join(ROOT, "data/era/day2/ground_truth.json")))
    names = {a["Id"]: a["Name"] for a in gt1["accounts"]}
    enc = Encoder(len(c1.vocab) + 1)
    enc.load_state_dict(torch.load(f"{OUT}/enc_good_s0.pt"))
    tok = lambda texts: [np.array([c1.vocab.get(t, 0) for t in tokenize(s)][:96], dtype=np.int64) for s in texts]  # noqa: E731
    STORE = os.environ.get("STORE", "jepa_good")
    if STORE in ("minilm", "dual", "route"):
        from sentence_transformers import SentenceTransformer as _ST
        _m = _ST("sentence-transformers/all-MiniLM-L6-v2", cache_folder=os.path.join(ROOT, "models"), device="cpu")
        mini = lambda texts: _m.encode(list(texts), batch_size=128, show_progress_bar=False, convert_to_numpy=True,  # noqa: E731
                                       normalize_embeddings=True)
        cache = os.path.join(OUT, "minilm_day1.npy")
        Xm = np.load(cache) if os.path.exists(cache) else mini([o["text"] for o in c1.obs])
        if STORE in ("minilm", "route"):
            emb_texts, X = mini, Xm
        else:   # two channels: the good-view state vector and the pretrained detail vector, side by side
            l2 = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)  # noqa: E731
            emb_texts = lambda texts: np.concatenate([l2(enc.embed(tok(texts))), l2(mini(texts))], 1)  # noqa: E731
            X = np.concatenate([l2(np.load(f"{OUT}/emb_good_s0.npy")), l2(Xm)], 1)
    else:
        emb_texts = lambda texts: enc.embed(tok(texts))  # noqa: E731
        X = np.load(f"{OUT}/emb_good_s0.npy")
    assert len(X) == len(c1.obs)
    results["setup"]["store_encoder"] = STORE
    f = fit_store_map(X)
    sem = f(X)
    n_sys = len(c1.systems)
    S = rows(sem, c1.ts, c1.system, n_sys)
    d_content = S.size(1)
    if STORE == "route":     # retrieval keys from the good-view state channel, appended after the content
        Xk = np.load(f"{OUT}/emb_good_s0.npy")
        fk = fit_store_map(Xk)
        key_texts = lambda texts: fk(enc.embed(tok(texts)))  # noqa: E731
        S = torch.cat([S, rows(fk(Xk), c1.ts, c1.system, n_sys)], 1)

    qs = make_questions(gt1, c1.states)
    rng = np.random.default_rng(0)                          # the same split as run_memory.py
    states = sorted({(q["account"], q["quarter"]) for q in qs})
    perm = rng.permutation(len(states))
    split = {states[i]: ("train" if k < 0.7 * len(states) else "val" if k < 0.8 * len(states) else "test")
             for k, i in enumerate(perm)}
    for i, q in enumerate(qs):
        q["split"], q["qi"] = split[(q["account"], q["quarter"])], i
    train_q = [q for q in qs if q["split"] == "train"]
    val_q = [q for q in qs if q["split"] == "val"]
    test_q = [q for q in qs if q["split"] == "test"]
    Q = rows(f(emb_texts([q["text"] for q in qs])), np.array([q["ts"] for q in qs]), None, n_sys)
    acct_texts = [f"What happened with {names.get(a, a)}?" for a in c1.accounts]
    acct_q = f(emb_texts(acct_texts))
    make_query = None
    if STORE == "route":
        Q = torch.cat([Q, rows(key_texts([q["text"] for q in qs]), np.array([q["ts"] for q in qs]), None, n_sys)], 1)
        acct_k = key_texts(acct_texts)

        def route_query(a, t):
            tc = time_code(np.array([t]), 16)[0]
            z = np.zeros(n_sys, np.float32)
            return np.concatenate([acct_q[a], tc, z, acct_k[a], tc, z]).astype(np.float32)
        make_query = route_query

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=os.path.join(ROOT, "models"), device="cpu")
    bank = torch.from_numpy(st.encode([a for k in ANSWERS for a in ANSWERS[k]], convert_to_numpy=True,
                                      normalize_embeddings=True))
    kind_ix, o = {}, 0
    for k in ANSWERS:
        kind_ix[k] = torch.arange(o, o + len(ANSWERS[k])); o += len(ANSWERS[k])

    # day2: grow the store (old rows untouched), questions about the new quarter
    c2 = load(D2, vocab=c1.vocab, accounts=c1.accounts)
    seen = {o["src_id"] for o in c1.obs}
    new = [i for i, o in enumerate(c2.obs) if o["src_id"] not in seen and c2.account[i] >= 0]
    sys2 = np.array([c1.systems.index(c2.obs[i]["system"]) for i in new])
    Xnew = emb_texts([c2.obs[i]["text"] for i in new]) if STORE in ("minilm", "dual", "route") else enc.embed([c2.tok[i] for i in new])
    new_rows = rows(f(Xnew), c2.ts[new], sys2, n_sys)
    if STORE == "route":
        new_rows = torch.cat([new_rows, rows(fk(enc.embed([c2.tok[i] for i in new])), c2.ts[new], sys2, n_sys)], 1)
    S2 = torch.cat([S, new_rows])
    q_new = make_questions(gt2, sorted({s for s in c2.states if s[1] in ("2026Q4", "2027Q1")}))
    for i, q in enumerate(q_new):
        q["qi"] = i
    Qn = rows(f(emb_texts([q["text"] for q in q_new])), np.array([q["ts"] for q in q_new]), None, n_sys)
    if STORE == "route":
        Qn = torch.cat([Qn, rows(key_texts([q["text"] for q in q_new]), np.array([q["ts"] for q in q_new]), None, n_sys)], 1)
    maj_new = float(np.mean([q["answer_ix"] == max(set(a := [x["answer_ix"] for x in train_q if x["kind"] == q["kind"]]),
                                                   key=a.count) for q in q_new]))
    results["growth_setup"] = {"new_records": len(new), "new_questions": len(q_new), "majority_on_new": maj_new}
    log("growth setup", results["growth_setup"])

    runs = results.setdefault("runs", {})
    for seed in SEEDS:
        grid = [("closed_book", "none", False), ("memorilla_no_pretrain", "memorilla", False),
                ("top16_pretrained", "topk", True), ("memorilla_pretrained", "memorilla", True)]
        only = os.environ.get("VARIANTS")
        for name, variant, pre in grid:
            if only and name not in only.split(","):
                continue
            t = time.time()
            torch.manual_seed(seed)
            mem = Memory(d_content, bank.size(1), variant=variant, split=d_content if STORE == "route" else None)
            if pre:
                pretrain_memory(mem, S, sem, c1.ts, c1.account, c1.system, n_sys, acct_q, steps=PRE, seed=seed, log=log,
                                make_query=make_query)
            mem = train_memory(S, Q, train_q, bank, kind_ix, variant, epochs=EPOCHS, batch=32, seed=seed, log=log,
                               val=val_q, init=mem)
            r = evaluate(mem, S, Q, test_q, bank, kind_ix)
            if variant != "none":
                r["growth"] = {"stale_store": evaluate(mem, S, Qn, q_new, bank, kind_ix)["overall"],
                               "grown_store": evaluate(mem, S2, Qn, q_new, bank, kind_ix)["overall"]}
            else:
                r["growth"] = {"closed_book": evaluate(mem, S, Qn, q_new, bank, kind_ix)["overall"]}
            r["seconds"] = time.time() - t
            runs[f"{name}/s{seed}"] = r
            log(name, seed, json.dumps(r))
            json.dump(results, open(f"{OUT}/memory2{SUFFIX}.json", "w"), indent=1)
            if name == "memorilla_pretrained" and seed == SEEDS[0]:
                explain(mem, S2, Qn, q_new, c1, c2, new, results)
                json.dump(results, open(f"{OUT}/memory2{SUFFIX}.json", "w"), indent=1)
    log("done")


@torch.no_grad()
def explain(mem, S2, Qn, q_new, c1, c2, new, results):
    obs_all = c1.obs + [c2.obs[i] for i in new]
    acct_all = np.array([o.get("account_id") for o in obs_all])
    mem.eval()
    ex = []
    for q in q_new[:: max(1, len(q_new) // 6)][:6]:
        _, attn = mem(S2, Qn[[q["qi"]]], store_raw=S2, q_raw=Qn[[q["qi"]]], return_attn=True)
        a = attn[-1][0].mean((0, 1))
        top = torch.topk(a, 6).indices.tolist()
        ex.append(dict(question=q["text"], answer=q["answer"],
                       attention_on_this_customer=float(a[torch.from_numpy(acct_all == q["account"])].sum()),
                       share_of_store_that_is_this_customer=float((acct_all == q["account"]).mean()),
                       read=[f"{obs_all[i]['system']} {quarter(obs_all[i]['ts'])} [{obs_all[i].get('account_id')}]: "
                             f"{obs_all[i]['text'][:80]}" for i in top]))
    results["attention_examples"] = ex


if __name__ == "__main__":
    main()
