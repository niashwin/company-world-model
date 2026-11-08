# Responses to the reviews

A note on terms: versions 2 to 4 sometimes called good views "link-trained views". From version 5, the paper, the
README and the website use "good views" for the pairs that the links give, and "good-view space" for the space that
an encoder learns from them. The responses below keep the terms of their time. [`CHECKS.md`](CHECKS.md) records the
checks of version 5.

- [The "corporate AI" review](#response-to-the-corporate-ai-review) → version 4
- [Review of version 2](#response-to-the-review-of-version-2) → version 3
- [Review of version 1](#response-to-the-review-of-version-1) → version 2

# Response to the "corporate AI" review

The review asked for four things: action-conditioned dynamics, counterfactual decision simulation, the two-channel
test on queries without names, and an abstract framed around corporate AI. Version 4 does all four on the same
Northwind data, with no new simulator runs (`cwm/run_actions.py`, `results/actions.json`, §6.11 and Tables 12–13).
The results are reported as they came out, after an independent check of the claims against the results. They do
not show that conditioning on logged actions unlocks dynamics. They do show, at least in part, why it doesn't, and
that points to what a decision layer needs.

## 1. Action-conditioned dynamics, s(t+1) = h(s(t), a(t))

**Done.** For each customer and month we read six actions from the records and Era's ground truth:

| Action | Share of customer-months |
| --- | --- |
| Sales call held | 21% |
| A leader (CEO or head of engineering) writing to or about the customer | 71% |
| CRM task logged | 56% |
| Engineering escalation (a Jira issue for the customer) | 15% |
| Support agents replying on the customer's tickets | 43% |
| Employees writing on the customer's email threads | 78% |

The ridge predictor of Table 10 was then conditioned on these actions (leave one customer out, ridge strength tuned by
nested validation), in four representations: link-trained views, bad views, the pretrained encoder and both channels.
Error changes below are shares of the state-only predictor's error; positive means worse.

**Result.**
- **This month's actions add nothing.** The error changes by −0.1% to +0.6% (link-trained views: +0.4%, 95% interval
  −1.6% to +2.3%). Shuffling the actions across months costs 1.0–1.4%.
- **In two of the four spaces, the state already holds them.** A logistic regression on the month's state predicts
  each action with AUROC 0.63–0.88 for link-trained views and 0.77–0.98 for bad views. In the pretrained and
  two-channel spaces they are harder to read (0.49–0.86 and 0.45–0.88) and still add nothing, so redundancy is not
  the whole explanation.
- **Next month's actions help, mostly as a description of next month's records.** They lower the error by 9–19%
  (link-trained views: 8.7%, interval 6.9% to 10.7%). But a Jira record is an escalation, and each other action is a
  kind of record of one system. Next month's mix of systems lowers the error by about as much on average (10–24%),
  though unevenly across customers (every interval includes zero). With the mix in the model, next month's actions
  add 3–5% (the interval excludes zero only for link-trained views).
- **Outcomes: actions add signal, mostly through ticket persistence.** Adding the month's actions raises next-month
  AUROC by +0.02 to +0.07 for deal progress (the interval excludes zero in one of four representations) and +0.02 to
  +0.26 for urgent tickets (three of four). Adding support replies alone does as well as all six actions
  (differences −0.02 to +0.02, every interval including zero). A support reply marks a month with a ticket (99% of
  months with a ticket have one, 6% without), and the indicator of any ticket this month ranks next month's urgent
  tickets as well as the reply indicator does (AUROC 0.79). The reply mostly marks that persistence rather than the
  effect of a decision.
- **Event windows** (14 days before and after each action, with all records of the action's kind left out):
  - around engineering escalations the state moves more than at matched times (+0.011, interval +0.004 to +0.020),
    but those windows hold more records (median 64 against 46). With record counts accounted for, the difference is
    +0.009 (interval −0.003 to +0.017), no longer detectable;
  - around sales calls it is 0.000 (−0.008 to +0.010), and around leaders' messages −0.024 (−0.065 to +0.022);
  - knowing which action was taken changes the error of a linear predictor by +0.02%.

So the missing action term was not why dynamics failed in Table 10.

## 2. Counterfactual decision simulation

**Done** (Table 13). For the 29 customer-months in trouble (an urgent or high ticket during an open deal), we
compare "intervene" (a sales call or engineering escalation that month) with "do not" in four ways:
- the outcomes observed in those months;
- case-based simulation: the most similar states of other customers (up to 15 per decision, from the 60 nearest),
  split by what the company did, in each of the four state spaces;
- all other customers' months, without matching on state;
- an outcome model with the decision as an input.

Intervals resample customers, repeating the neighbour search and refitting the outcome model.

**Result.** The machinery works, but the answers are not effects:
- **The logs barely contain the alternative.** The company intervened in 25 of the 29 troubled months, which come
  from eight customers (12 from one).
- **Decisions follow situations that the state does not fully capture.** The month's state predicts whether the company
  intervened (AUROC 0.68–0.94). Yet among the most similar states, those where it intervened were themselves in
  trouble 19–24% of the time and had an urgent ticket 57–62% of the time, against 0.2–4% and 17–35% where it did
  not, in all four state spaces.
- **So the comparison sets troubled months against calm ones.** Intervening appears to be followed by more urgent
  tickets (+29 points with link-trained views, interval +6 to +45). That is what one would see if calls and
  escalations are made when problems are severe, whatever their effect.
- **The estimates are not stable.** They move with the state space (+12 to +15 points for deal progress, +8 to +29 for
  urgent tickets, six of eight intervals include zero). The outcome model attributes little to the decision (0 and
  +5 points).
- **A monthly indicator mixes cause and response.** The first call or escalation came before the month's first urgent
  ticket in 13 of the 25 months and after it in 12.

The paper argues that learning from decisions needs actions whose assignment is known: decisions logged with their
reasons, deliberately varied, or replayed in a simulator.

## 3. The two-channel store on queries without names

**Done in version 3** (Table 8). Each question comes with one record of its customer instead of the name.
- **Names kept.** Placing that record with the state channel and reading its customer's records answers 63.8%,
  against 54.9% with the pretrained encoder. The paired difference is +8.9 points (95% interval +3.9 to +13.8).
- **Names masked.** Only the record's own link helps clearly.

The figures the review predicted (53% vs 76%) are held-out placement accuracies from Table 3, not answer accuracies.

## 4. Framing around corporate AI

**Done.** The abstract, introduction and §7 now say that corporate AI is missing an operational world model.
- **Training economics, as measured.** Every encoder and head we timed trains in 2–7 minutes on a CPU, without a GPU,
  including each reader's encoder. At this scale a company could retrain its state space and each access scope's
  model daily, or whenever an access rule changes; a new §7 paragraph covers this.
- **What we did not claim.** We did not measure what fine-tuning a large language model on the same data costs, so
  the paper says only that it would typically need GPUs. Whether training stays this cheap with millions of records
  is untested.
- **The active thesis follows the data:**
  - the state space works;
  - memory should route through it before reading;
  - the decision layer is built and tested; on logged data it learns little about what decisions do, and we argue
    it needs actions whose assignment is known.

We did not write that action-conditioned transitions unlock decision learning, because on this data they did not.

# Response to the review of version 2

The review asked for one more experiment, questions where the state channel is actually needed, and for the
abstract to present the negative results as boundary conditions. Both are done in version 3. The experiment gives a
clear positive result with names kept and a clear null with names masked, though not exactly the one the reviewer
predicted, and it sharpens the "division of labor". Numbers come from `results/memory_anchor_*.json`: four seeds
each, 127 test questions, with paired bootstrap intervals over questions.

## 1. The "smoking gun" for the routed two-channel store

**Asked.** Evaluate the same questions where customer names are stripped from the stored records, and show that
routing by the state channel and reading the detail channel beats both single channels.

**Why the experiment needed a different shape.** If the stored records are masked but the question still names the
customer, neither channel can match the name, so no store can find the customer. In version 3 the question
therefore no longer names the customer. It comes instead with **one record of the customer (an anchor)**, as if
asked by someone holding a support ticket: "was there a sales call with this customer this quarter?" The anchor
comes from a thread hidden from the state encoder's training and the reader's pre-training, and from a different
system than the one that answers the question. We ran it with every customer name, employee name and identifier
masked, and with names kept (`cwm/run_memory_anchor.py`, Table 8 and §6.8 of the paper). A store finds 16 records of
the question's quarter in one of three ways:
- the anchor's nearest neighbours, by the content of one channel;
- **placement**: the records of the customer whose centroid in one channel is nearest to the anchor;
- an oracle that uses the anchor's own link.

**Result** (accuracy; in brackets, the share of the 16 records found that belong to the anchor's customer):

| How the records are found | Reading | Names masked | Names kept |
| --- | --- | --- | --- |
| No store | | 52.0% | 52.2% |
| Neighbours, detail channel | detail | 54.7% (38%) | 58.3% (53%) |
| Neighbours, state channel | detail | 57.7% (36%) | 60.0% (67%) |
| Placement, detail channel | detail | 52.2% (18%) | 54.9% (41%) |
| **Placement, state channel** | detail | 52.8% (25%) | **63.8% (66%)** |
| Placement, both concatenated | detail | 55.1% (25%) | 64.2% (66%) |
| Anchor's own link (oracle) | detail / state / system and time only | 67.1% / 69.9% / 68.1% | 68.7% / 69.9% / 67.5% |

- **Names kept: finding with the state channel wins.** State placement beats detail placement by **+8.9 points**
  (95% interval +3.9 to +13.8; positive in every seed) and no store by +11.6. It comes within 5 points of the oracle.
  Neighbours point the same way less clearly (60.0% vs 58.3%). Placement reads a sample of the whole situation, which
  more often contains a record from the system that answers the question (9.8% of records read, against 6.5% for the
  state channel's neighbours and 3.2% for the detail channel's).
- **Names masked: one record rarely finds its customer.** The state channel still places masked held-out records
  with the right customer more often (33% vs 20%). Its neighbours help a little: +5.7 points over no store, with a
  95% interval of +0.4 to +10.8. Placement does not help (+0.8), and only the oracle helps clearly.
- **Reading matters little; finding matters.** Given the oracle's records, reading only each record's system and
  time codes (67.5–68.1%) does about as well as reading its content (67.1–69.9%). For these counting questions, the
  provenance of the right records carries most of the answer.

**What this does and does not support.**
- *Supported:* "find a question's records through the state space, then read them from the store."
- *Not supported:* "routing *rather than blending*". Placing with the two channels concatenated works as well as the
  state channel alone. The paper says so.

**What an independent check changed.** Before release, a separate check of the first draft of this experiment found
real problems, and all of them are fixed:
1. *Time-dominated retrieval keys.* The neighbour stores ranked whole store rows, in which a time code outweighed the
   content. That handicapped them, and it also affected the named-question test (Table 7): only 24–35% of retrieved
   records belonged to the asked customer. Retrieval now ranks records of the question's quarter by content alone
   (65–71% right customer), and Table 7 was rerun: detail only 71.2%, state only 65.8%, concatenated 72.2%, routed
   70.1%. The conclusion is unchanged; no combination clearly beats the detail channel.
2. *Pre-training links.* The reader's pre-training had used links of the held-out threads. It now uses training
   threads only.
3. *Untested provenance claim.* The claim that provenance carries the answer was untested. The provenance-only
   oracle now tests it.
4. *Single-number comparisons.* Comparisons are now paired over questions, with bootstrap intervals.

The first, flawed draft (time-dominated keys, plus a variant restricted to other systems) is not in the paper.

## 2. Reframe the abstract as boundary conditions

Done. The abstract, introduction, §7 and conclusion now present the results as a **division of labor** among the
three parts of a world model:
- **State space:** learned from links alone; places records without links; shared across encoders and versions
  through rotations; carries access control.
- **Memory:** find a question's records through the state space, or by name when the question has one, then read
  them from the store, whose records supply what they say and where they came from.
- **Dynamics:** did not emerge here from a space trained for static similarity. Forecasts, linear dynamics, drift
  and cross-team distances stay at chance or at simple baselines. We expect a predictor to need training on
  transitions and actions, probably across many companies.

We did not call it a "separation theorem". These lines are drawn on one simulated company, and the paper says they
may move with more data.

## Smaller corrections to the review's summary

- The 398 monthly transitions are counted through day 2 (469 states); day 1 has 405 month states.
- The "rotat rotation" typo was not in the source; version 2 already stops the word from being hyphenated.

# Response to the review of version 1

Version 2 of the paper ([`paper/main.pdf`](../paper/main.pdf)) runs every experiment the review asked for, on the
same simulated company (Northwind Robotics, Era by Eon). Some results confirm what version 1 proposed; several do
not, and the paper now says so plainly with the controls that make the negative results interpretable. Numbers
below come from `results/*.json` and appear in the paper; one seed unless noted.

## Summary

| Review item | What we ran | Outcome |
| --- | --- | --- |
| 1. Two-channel store (state + detail) | Concatenated and routed stores on the 144 questions (4 seeds each); two-channel state probes | **Not confirmed.** Neither combination beats the detail channel alone (70.8% detail only, 69.6% concatenated, 69.3% routed, 66.1% state only). The detail channel restores system facts in state probes (call 72% → 87–95%). |
| 2. Rescue low-visibility readers without leaking | Shared pretrained backbone + per-reader heads, a public-records base, both channels per reader, a scope diagnostic; rotations analysed | **Partly.** Financial analyst 23% → 43% with zero leakage and exact non-interference; software engineer 28% → 32% at best. Rotations cannot raise accuracy by construction. |
| 3. Finer time windows | Monthly states (405 on day 1; 469 states and 398 transitions through day 2); weekly states for drift | **Mostly negative.** Months are told apart a little better than a strong reference (17% vs 13%); no forecast, no linear dynamics beyond shrinkage to the mean, no drift signal. |
| 4. Operational primitives | Cross-team misalignment, similar situations across customers (with time and "trouble" controls) | **Misalignment: no signal. Similar situations: about twice chance**, and only the detail channel finds other customers in trouble. |
| 5a. Mask employee names | Customers and employees masked, encoder retrained | **Confirmed.** 47% vs 48% with customers only: employee names carried almost none of the remaining signal. |
| 5b. Pretrained baseline in Tables 2 and 3 | Frozen all-MiniLM-L6-v2, out of the box and with a good-view head | **Confirmed.** 40% → 72% (training records), 53% → 62% (held-out); below the scratch encoder (95%, 76%). |
| Typo "rotat rotation" | | Not in the source; the word is now excluded from hyphenation so a PDF extractor cannot split it. |

## 1. Build and test the two-channel architecture

**Asked.** Combine the good-view state vector with a detail representation and show that the combined store
beats both the pretrained-only store and the JEPA-only store on the 144 test questions, and fixes the counting
facts (call: 72% for good views vs 97% for bad views).

**Done.**
- *Memory store* (`cwm/run_memory2.py`, Table 7 of the paper): four stores, same reader, four seeds each.
  - state channel only (good views): **66.1% ± 2.0**
  - detail channel only (pretrained encoder): **70.8% ± 0.0** (the four runs happen to get 102/144 right, with
    different answers)
  - both channels concatenated (`STORE=dual`): **69.6% ± 3.8**
  - both channels routed, retrieving with the state channel and reading the detail channel (`STORE=route`):
    **69.3% ± 1.6**
- *State probes* (`run_v2.py state`, Table 6): good views alone read "was there a call this quarter" 72% of the
  time (about the majority class); with the pretrained encoder as the detail channel 87%, with bad views 95%.
  Averaged over the four facts, the best single channel (bad views, 70%) still beats every combination
  (good + bad 67%, good + pretrained 62%).

**Finding.** The detail channel restores the facts the state channel discards, but no way of combining the two
beats the detail channel alone on these questions. The reason is visible in the benchmark: all 144 questions
(ticket counts, worst priority, whether a call happened, deal stage) are about counting or reading one system's
records, which is exactly what the detail channel keeps and the state channel is trained to ignore. The state
channel's strength, placing records by situation across systems (76% vs 53% for the pretrained encoder on held-out
records), is not what these questions test. The paper (§6.8) now presents the two channels as doing different jobs
and recommends serving each question from the channel suited to it, and lists a benchmark with questions that need
the state channel (answers in unlinked records) as a limitation (§9).

## 2. Rescue low-visibility roles under access control

**Asked.** Use a shared base encoder or orthogonal Procrustes rotations to lift the software engineer (28%) and
financial analyst (23%) toward 60% while keeping zero leakage.

**The constraint that decides what is possible.** Non-interference means everything a reader receives may depend
only on records they may read, plus knowledge that contains no company data. So:
- a **pretrained backbone** (trained on public text) may be shared by everyone;
- a **base trained on records every employee may read** may be shared by everyone;
- a **rotation** between two scopes' spaces is an isometry: it changes the coordinates of a reader's vectors and
  preserves all distances between them, so nearest-centroid placement and search are exactly as accurate after it
  as before. It could help only by bringing in another scope's states or index, which non-interference forbids,
  and a rotation toward a space the reader may not read is itself fitted with that scope's encoder, so it is
  allowed only toward spaces the reader may read (such as the public base) or for someone entitled to both scopes.
  Rotations make views *comparable*; they cannot make a view *more accurate*. A unit test
  (`tests/test_smoke.py::test_rotation_does_not_change_placement`) checks the accuracy part.
- the 66% / 65% of the encoder trained on everything is **not a legitimate target**: part of its advantage comes
  from records the reader may not read, which is precisely what the planted secrets showed leaking.

**Done** (`run_v2.py acl acl_combo acl_pub scope`, Table 5 and Figure 8):

| Reader (share readable) | Own encoder | Backbone, no head | Backbone + own head | Both side by side | Public base, fine-tuned | Trained on everything (leaks) |
| --- | --- | --- | --- | --- | --- | --- |
| Software engineer (22%) | 28% | 15% | 29% | **32%** | 25% | 66% |
| Financial analyst (9%) | 23% | 35% | **43%** | 30% | 41% | 65% |
| CEO (100%) | **76%** | 53% | 62% | | | 76% |

- **Leakage stays at zero.** With per-reader heads on the shared backbone, the codename and secret-sentence
  advantage (share found with the secrets minus share found without) is +0.00 for both readers, and their
  embeddings are identical with and without the planted secrets (largest difference 0). A head trained on
  everything also leaks through filtered search: codename advantage +0.28 and +0.40, secret-sentence advantage
  +0.60 and +0.60 (scratch encoder trained on everything: +1.00 and +0.40 for both readers). The financial analyst
  may read the CRM records carrying the secrets, so the leak test does not apply to that reader.
- **Public base.** Only 534 linked records may be read by every employee. A head trained on them places 17% and 38%
  of the two readers' records; fine-tuned on each reader's records for as many steps as a head trained directly, 25%
  and 41%, below the head trained directly (29% and 43%). It adds nothing here.
- **Why the two readers differ.** 87% of what the software engineer may read is support tickets, which the backbone
  alone places poorly (15%, against 35% for the analyst's CRM records and emails).
- **Number versus kind of records.** The same scratch encoder trained on a random sample of all systems, as many
  records as each reader may read, places 42% of the software engineer's records and 31% of the financial
  analyst's. Most of the gap to the encoder trained on everything is records the reader may not read at all; a
  smaller part is that their records come from one or two systems.

**Finding.** A shared backbone rescues one low-visibility reader substantially (23% → 43%) and the other barely
(28% → 32%), with the guarantee intact. Reaching ~60% would require information the reader is not entitled to; that
is a decision about the access policy (for example, a customer summary everyone may read), not about the model. The
paper says this in §4 ("Sharing what may be shared") and §6.7.

## 3. Finer temporal granularity

**Asked.** Re-slice into monthly or bi-weekly windows to multiply transitions, then show a linear dynamics model or
drift detector flagging the week a trajectory shifts toward an escalation or deal-stage change.

**Done** (`run_v2.py monthly time primitives`, Tables 8–10):
- *Months.* 405 customer-month states on day 1; 469 states and 398 month-to-month transitions through day 2
  (2.4× the quarterly transitions). ISO weeks: 1,445 customer-weeks with enough records, 279 of them with an
  urgent/high ticket or a deal-stage change.
- *Telling months apart.* A good-view encoder trained on month states places held-out records in the right
  customer-month 17% of the time, against 13% for "customer known, its busiest month" (with quarters, good views
  only matched the busiest-quarter reference: 26% vs 26%).
- *Forecasting.* Next-month facts: 46–54% for every representation against 54% for the majority class and 72% for
  copying this month's facts.
- *Linear dynamics* (ridge from a state to its change, leave one customer out). With a fixed weak regularization
  (α = 10) it overfits and is worse than predicting no change in the good-view space (+55% squared error). With α
  chosen by nested leave-one-customer-out validation it beats "no change" in every space (2–31% less error), but
  never by more than 3.4 points beyond a single shrink-toward-the-mean factor, itself a linear predictor, and in four
  of six spaces by nothing: the model learns regression to the mean and almost nothing more.
- *Drift.* Weekly drift scores flag escalation or deal-change weeks, or the week before, with AUROC 0.47–0.54
  (two standard errors ±0.04); record volume alone reaches 0.57.

**Finding.** Finer windows help the model tell a customer's months apart a little, but do not unlock forecasting,
dynamics or drift detection on this data. Possible reasons, which we cannot separate here: Era may generate
escalations without precursors in other systems' text; weekly and monthly states average few records; and the
good-view objective makes states stable by design rather than sensitive to change (§6.9, §6.10).

## 4. Operational primitives

**Asked.** (a) Show that the distance between a customer's sales/CRM state and support/engineering state rises with
real escalations or deal risk. (b) Show that a troubled customer's state retrieves other customers in the same
trouble.

**Done** (`run_v2.py primitives situations`, Table 10):
- *Misalignment.* 163 customer-months with at least two sales records (Salesforce, Gong) and two support records
  (Zendesk, Jira); 31 at risk (urgent/high ticket during an open deal). AUROC 0.48–0.57 across representations,
  within two standard errors of chance (±0.12). **No signal.**
- *Similar situations.* For 160 customer-quarter states, the five most similar states of *other* customers share the
  situation (ticket bucket × deal-stage group) 18% of the time with good views (21% bad views, 18% pretrained)
  against 8% chance. Controls: restricted
  to other customers' states in *other* quarters, 17% vs 7%; picking other customers in the same quarter gives 13%.
- *Customers in trouble* (22 quarters with an urgent/high ticket during an open deal): among the five nearest
  other-customer states, 35% are also in trouble with bad views and 25% with the pretrained encoder, against a 12% base
  rate, but only 4% with good views.

**Finding.** Similar-situation search works at about twice chance with every representation and survives the time
control; finding other customers *in trouble* needs the detail channel, because trouble rests on facts about
particular systems' records (ticket priorities, deal stages). Misalignment shows nothing on this data.

## 5. Methodological controls

**Employee masking** (`run_extras.py extras_e maskall:good`, §6.3). In addition to customer names, contacts,
domains, email addresses and identifiers, we masked every employee's name words, email handle and chat names (18,936 linked
records change, against 18,832 with the customer mask alone, because most records that name an employee were
already changed). Same-customer cross-system precision: 47% (customers masked only: 48%; TF-IDF 14%);
state-level precision 25%, still above the 16.3% that knowing only the customer would give. The residual half of
the signal is not employee identity.

**Pretrained baseline in Tables 2 and 3** (`run_v2.py pretrained`, `cwm/pretrained.py`). Frozen all-MiniLM-L6-v2,
out of the box and with a small JEPA head (LayerNorm → 384→512→256) trained on good views:

| | Same customer, other systems (training records) | Held-out: right customer |
| --- | --- | --- |
| Pretrained, frozen | 40% | 53% |
| Pretrained + good-view head | 72% | 62% |
| Good views, scratch encoder | 95% | 76% |
| TF-IDF | 35% | 53% |

The link-supervision recipe improves an off-the-shelf backbone substantially. On this company, a frozen
general-purpose backbone with a head stays below the small scratch encoder, which learns the company's own
vocabulary.

## Housekeeping

- "rotat rotation" does not occur in the LaTeX source ("one rotation fitted on 64 anchor pairs"); it is most likely
  an artifact of extracting a hyphenated line break from the PDF. Version 2 adds `\hyphenation{rotation rotations}`
  so the word is never split.
- Every number in the paper is filled from `results/` by `cwm/fill_paper.py`; the build fails loudly if a
  placeholder is left unfilled.
- New tests: the good-view head trains on frozen features, and placement is invariant to rotations.

## What did not change

The core claims of version 1 stand: good views from cross-system links organize the space by situation, generalize
to held-out records, survive masking, degrade gracefully with wrong links, align partly across separately trained
encoders by one rotation, and let a retrained encoder search the old index; access control built into the model
closes the three leak paths with exact non-interference.
