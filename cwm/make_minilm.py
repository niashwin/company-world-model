"""Cache frozen all-MiniLM-L6-v2 features for every linked record of the day-1 and day-2 corpora.

Writes results/minilm_day1.npy and results/minilm_day2.npy (rows in corpus order, L2-normalized). Day-1 records are
also in the day-2 export with the same source id and text, so their features are taken from the day-2 pass; any
day-1 record missing from day 2 is embedded separately. About ten minutes on a CPU.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load  # noqa: E402
from pretrained import minilm_embed  # noqa: E402
from run_prh import D1, D2, OUT  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    c1 = load(D1)
    c2 = load(D2, vocab=c1.vocab, accounts=c1.accounts)
    F2 = minilm_embed([o["text"] for o in c2.obs])
    idx2 = {o["src_id"]: i for i, o in enumerate(c2.obs)}
    miss = [i for i, o in enumerate(c1.obs) if o["src_id"] not in idx2]
    F1 = np.zeros((len(c1.obs), F2.shape[1]), np.float32)
    for i, o in enumerate(c1.obs):
        if o["src_id"] in idx2:
            F1[i] = F2[idx2[o["src_id"]]]
    if miss:
        F1[miss] = minilm_embed([c1.obs[i]["text"] for i in miss])
    np.save(os.path.join(OUT, "minilm_day1.npy"), F1)
    np.save(os.path.join(OUT, "minilm_day2.npy"), F2)
    print("done", F1.shape, F2.shape, "day-1 records missing from day 2:", len(miss), f"{time.time() - t:.0f}s", flush=True)
