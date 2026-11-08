"""Count the records of the simulated company (Extended Data Table 1 and the data paragraph of Methods).

Reads data/era/day1 and data/era/day2 (written by cwm/era_export.py) and writes results/corpus.json:
records and linked records per system on day 1, customers, customer-quarter states, and the linked records that
day 2 adds. "Records" counts every exported record. A record is linked if its system ties it to a customer; the
experiments use the linked records that have a time and a text (cwm/data.py), which on day 1 is all of them.
New records on day 2 are linked records of the day-1 customers whose source ID is not among the day-1 linked
records (as in cwm/run_prh.py, experiment E3).
"""
import json
import os
from collections import Counter

from data import window_key
from run_prh import D1, D2, OUT


def records(path):
    return [json.loads(line) for line in open(path)]


def usable(o):
    return bool(o.get("ts") and o.get("text") and o.get("account_id"))


def main():
    day1, day2 = records(D1), records(D2)
    by_sys = Counter(o["system"] for o in day1)
    linked = Counter(o["system"] for o in day1 if o.get("account_id"))
    lk1 = [o for o in day1 if usable(o)]
    customers = sorted({o["account_id"] for o in lk1})
    states = {(o["account_id"], window_key(o["ts"], "quarter")) for o in lk1}
    seen = {o["src_id"] for o in lk1}
    new_linked = [o for o in day2 if usable(o) and o["account_id"] in set(customers) and o["src_id"] not in seen]
    quarters = sorted({window_key(o["ts"], "quarter") for o in lk1})
    out = {
        "day1": {"records": len(day1), "linked": sum(linked.values()),
                 "by_system": {s: {"records": by_sys[s], "linked": linked[s]} for s in sorted(by_sys)}},
        "customers": len(customers),
        "customer_quarter_states": len(states),
        "first_quarter": str(quarters[0]), "last_quarter": str(quarters[-1]),
        "day1_usable_linked": len(lk1),
        "day2": {"records": len(day2), "linked": sum(1 for o in day2 if o.get("account_id")),
                 "new_linked_known_customers": len(new_linked),
                 "new_linked_by_month": dict(sorted(Counter(str(window_key(o["ts"], "month")) for o in new_linked).items())),
                 "new_linked_later_by_system": dict(Counter(o["system"] for o in new_linked
                                                            if str(window_key(o["ts"], "month")) > "2027-01"))},
    }
    json.dump(out, open(os.path.join(OUT, "corpus.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
