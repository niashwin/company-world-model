#!/usr/bin/env bash
# Reproduce every result, figure and the PDF of the paper from nothing.
#
# Usage:  bash scripts/reproduce.sh [stage ...]
# Stages, in order:  data v1 v2 v3 v4 paper site      ("all" runs every stage; default: all)
#
#   data   start Era by Eon's seven simulators (Docker), export the day-1 and day-2 records to data/era/, and
#          count them                                                       -> results/corpus.json
#   v1     main comparison, alignment, new data, extra seeds, masking, wrong links, held-out records,
#          state tracking, access control, memory reader                    -> results/prh.json, extras_*.json, ...
#   v2     pretrained encoder, monthly windows, dynamics, primitives, access control with a backbone,
#          memory stores with two channels                                  -> results/v2_*.json, memory2_*.json
#   v3     named questions retrieved inside the quarter, and anchored questions -> results/memory2_q_*.json,
#                                                                              memory_anchor_*.json
#   v4     decisions: actions, action-conditioned dynamics, event windows, simulation -> results/actions.json
#   paper  figures and the PDF; every number comes from results/ through cwm/fill_paper.py
#   site   data and page of the project website (needs the caches of v1 to v4) -> website/data/site-data.js,
#                                                                              website/index.html
#
# Later stages read caches that earlier stages write to results/ (encoders *.pt, embeddings *.npy, *.npz).
# Runs are deterministic for a given seed and thread count, not across thread counts. The paper used one thread
# per process (THREADS=1 OMP_NUM_THREADS=1) on a machine with two cores. A full run takes several hours.
set -euo pipefail
cd "$(dirname "$0")/.."
export THREADS="${THREADS:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PY="${PY:-python}"
log() { printf '\n== %s ==\n' "$*"; }

stage_data() {
  log "data: start the simulators for day 1 and export"
  sh cwm/era_up.sh day1
  sleep 20
  $PY cwm/era_export.py --out data/era/day1
  log "data: move every simulator to day 2 and export again"
  for p in 8080 8081 8082 8083 8084 8085 8087; do curl -fsS --noproxy '*' -XPOST "localhost:$p/_admin/state/day2" >/dev/null; done
  $PY cwm/era_export.py --out data/era/day2
  $PY cwm/corpus_stats.py               # Extended Data Table 1 and the counts in Methods
  if ! git diff --quiet -- results/corpus.json 2>/dev/null; then
    echo "WARNING: the exported company differs from the paper's (results/corpus.json changed)."
    echo "         Era's images may have changed since November 2026. See REPRODUCE.md."
  fi
}

stage_v1() {
  log "v1: main comparison (Extended Data Table 2), alignment (Extended Data Fig. 3a), new data (Extended Data Fig. 4)"
  STEPS=1500 $PY cwm/run_prh.py e1      # good views, bad views, next-token prediction; caches encoders
  STEPS=1000 $PY cwm/run_prh.py e2      # separately trained encoders and one rotation
  STEPS=1500 $PY cwm/run_prh.py e3      # a new quarter of data
  $PY cwm/run_e3b.py                    # retraining: alignment and search of the old index
  log "v1: extra seeds, masking, wrong links, held-out records"
  $PY cwm/run_extras.py extras_a seeds:good,bad
  $PY cwm/run_extras.py extras_b mask:good noise:0.25,0.5 seeds:ntp
  $PY cwm/run_extras.py extras_c heldout:base,good,bad
  $PY cwm/run_extras.py extras_d heldout:ntp
  log "v1: state tracking, access control, memory reader"
  $PY cwm/run_state.py
  $PY cwm/run_acl.py train:open train:open_clean train:account_executive train:support_engineer \
                     train:support_engineer_clean train:software_engineer train:financial_analyst eval
  $PY cwm/run_memory2.py
  SEEDS=1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_seeds $PY cwm/run_memory2.py
}

stage_v2() {
  log "v2: customers and employees masked; features of the pretrained encoder"
  $PY cwm/run_extras.py extras_e maskall:good
  $PY cwm/make_minilm.py
  log "v2: pretrained encoder, months, access control with a backbone, dynamics, primitives"
  $PY cwm/run_v2.py embed2 pretrained monthly acl state time primitives situations scope acl_combo acl_pub
  log "v2: memory stores with the detail channel, both channels, and routing (four seeds each)"
  STORE=minilm SEEDS=0,1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_minilm4 $PY cwm/run_memory2.py
  STORE=dual   SEEDS=0,1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_dual    $PY cwm/run_memory2.py
  STORE=route  SEEDS=0,1,2,3 VARIANTS=top16_pretrained            SUFFIX=_route   $PY cwm/run_memory2.py
}

stage_v3() {
  log "v3: named questions, retrieval by content inside the question's quarter (Supplementary Table 2)"
  for s in jepa_good minilm dual route; do
    KEYS=quarter STORE=$s SEEDS=0,1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX="_q_${s#jepa_}" \
      $PY cwm/run_memory2.py
  done
  $PY cwm/memory_retrieval.py
  log "v3: questions that come with a record of the customer (Fig. 4 and Supplementary Table 3)"
  CORPUS=masked PREP=1 $PY cwm/run_memory_anchor.py
  CORPUS=plain  PREP=1 $PY cwm/run_memory_anchor.py
  CORPUS=masked $PY cwm/run_memory_anchor.py
  CORPUS=plain  $PY cwm/run_memory_anchor.py
}

stage_v4() {
  log "v4: decisions (Tables 12 and 13)"
  $PY cwm/run_actions.py table dynamics events simulate checks
}

stage_paper() {
  log "paper: figures and PDF"
  make all
}

stage_site() {
  log "site: data files of the project website"
  $PY cwm/export_site_data.py
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(all)
for s in "${stages[@]}"; do
  case "$s" in
    all) stage_data; stage_v1; stage_v2; stage_v3; stage_v4; stage_paper; stage_site ;;
    data|v1|v2|v3|v4|paper|site) "stage_$s" ;;
    *) echo "unknown stage: $s (use data v1 v2 v3 v4 paper site all)"; exit 2 ;;
  esac
done
log "done"
