"""Export one Era-by-Eon simulated company into a single observation table plus ground truth.

Every text-bearing record from every system becomes an *observation*:
    {obs_id, system, kind, ts, author, text, account_id, deal_id, thread, channel}
`account_id` is the ground-truth link the simulator's shared company graph provides (Slack account channels,
Gmail X-Account-Id, Gong CRM context, Zendesk organization, Jira customer field, Salesforce AccountId/WhatId).
It is used to define world states and to evaluate; the encoders never read it.

Usage:  python era_export.py --out data/era/day1     (simulators started by era_up.sh)
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import time
from datetime import datetime

import requests
from urllib.parse import urlparse

PORTS = dict(slack=8080, salesforce=8081, zendesk=8082, gong=8083, gmail=8084, calendar=8085, jira=8087)
H = {"Authorization": "Bearer era-local"}
S = requests.Session()
S.trust_env = False  # local simulators: bypass the outbound proxy


def url(system, path):
    return f"http://127.0.0.1:{PORTS[system]}{path}"


def get(system, path, **params):
    for attempt in range(4):
        r = S.get(url(system, path), headers=H, params=params, timeout=300)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1 + attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def path_of(next_url):
    """A vendor 'next page' link (any host) -> local path with its query string."""
    if not next_url:
        return None
    u = urlparse(next_url)
    return u.path + ("?" + u.query if u.query else "")


def post(system, path, body):
    r = S.post(url(system, path), headers=H, json=body, timeout=300)
    r.raise_for_status()
    return r.json()


def iso_ts(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("Z", "+00:00")
    s = re.sub(r"\.(\d{3})\+0000$", r".\1+00:00", s)
    s = re.sub(r"\+0000$", "+00:00", s)
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def clean(t):
    t = re.sub(r"<[^>]+>", " ", t or "")
    return re.sub(r"\s+", " ", t).strip()


def adf_text(node):
    """Atlassian document format -> plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(adf_text(n) for n in node)
    out = node.get("text", "")
    return (out + " " + adf_text(node.get("content", []))).strip()


# --------------------------------------------------------------------------------------------- salesforce
def soql(q):
    rows, res = [], get("salesforce", "/services/data/v60.0/query", q=q)
    rows += res.get("records", [])
    while not res.get("done", True) and res.get("nextRecordsUrl"):
        res = get("salesforce", res["nextRecordsUrl"])
        rows += res.get("records", [])
    for r in rows:
        r.pop("attributes", None)
    return rows


def export_salesforce(obs, gt):
    gt["accounts"] = soql("SELECT Id, Name, Industry, Type, OwnerId, NumberOfEmployees, Website, CreatedDate FROM Account")
    gt["opportunities"] = soql(
        "SELECT Id, Name, AccountId, StageName, Amount, CloseDate, OwnerId, IsClosed, IsWon, CreatedDate, Probability FROM Opportunity")
    gt["opportunity_history"] = soql(
        "SELECT Id, OpportunityId, StageName, Amount, CloseDate, CreatedDate FROM OpportunityHistory")
    gt["contacts"] = soql("SELECT Id, Name, Email, AccountId, Title FROM Contact")
    gt["sf_users"] = soql("SELECT Id, Name, Email, Title FROM User")
    for r in soql("SELECT Id, Subject, Status, Priority, AccountId, CreatedDate, Description, OwnerId FROM Case"):
        obs.append(dict(system="salesforce", kind="case", ts=iso_ts(r.get("CreatedDate")), author=r.get("OwnerId"),
                        text=clean(f"{r.get('Subject') or ''}. {r.get('Description') or ''}"), account_id=r.get("AccountId"),
                        thread=r["Id"], channel="case", src_id=r["Id"], meta=dict(status=r.get("Status"), priority=r.get("Priority"))))
    for r in soql("SELECT Id, Subject, Description, WhatId, WhoId, OwnerId, ActivityDate, CreatedDate, Status FROM Task"):
        what = r.get("WhatId") or ""
        obs.append(dict(system="salesforce", kind="task", ts=iso_ts(r.get("CreatedDate")), author=r.get("OwnerId"),
                        text=clean(f"{r.get('Subject') or ''}. {r.get('Description') or ''}"),
                        account_id=what if what.startswith("acct-") else None,
                        deal_id=what if what.startswith("deal-") else None, thread=r["Id"], channel="task", src_id=r["Id"]))
    for r in soql("SELECT Id, Subject, TextBody, FromAddress, ToAddress, RelatedToId, ParentId, MessageDate, CreatedDate FROM EmailMessage"):
        rel = r.get("RelatedToId") or ""
        obs.append(dict(system="salesforce", kind="crm_email", ts=iso_ts(r.get("MessageDate") or r.get("CreatedDate")),
                        author=r.get("FromAddress"), text=clean(f"{r.get('Subject') or ''}. {r.get('TextBody') or ''}"),
                        account_id=rel if rel.startswith("acct-") else None, deal_id=rel if rel.startswith("deal-") else None,
                        thread=r.get("ParentId") or r["Id"], channel="crm_email", src_id=r["Id"]))
    for r in gt["opportunities"]:
        obs.append(dict(system="salesforce", kind="opportunity", ts=iso_ts(r.get("CreatedDate")), author=r.get("OwnerId"),
                        text=clean(f"{r['Name']}. Stage {r.get('StageName')}. Amount {r.get('Amount')}. Close date {r.get('CloseDate')}."),
                        account_id=r.get("AccountId"), deal_id=r["Id"], thread=r["Id"], channel="opportunity", src_id=r["Id"]))


# --------------------------------------------------------------------------------------------- zendesk
def export_zendesk(obs, gt):
    orgs, page = [], "/api/v2/organizations.json?per_page=100"
    while page:
        res = get("zendesk", page)
        orgs += res["organizations"]
        page = path_of(res.get("next_page"))
    gt["zendesk_orgs"] = [{k: o.get(k) for k in ("id", "external_id", "name", "domain_names", "tags")} for o in orgs]
    org2acct = {o["id"]: o.get("external_id") for o in orgs}
    tickets, page = [], "/api/v2/tickets.json?per_page=100"
    while page:
        res = get("zendesk", page)
        tickets += res["tickets"]
        page = path_of(res.get("next_page"))
    gt["tickets"] = [{k: t.get(k) for k in ("id", "external_id", "status", "priority", "type", "organization_id", "created_at",
                                             "updated_at", "tags", "subject")} for t in tickets]

    def comments(t):
        try:
            return t, get("zendesk", f"/api/v2/tickets/{t['id']}/comments.json").get("comments", [])
        except Exception:
            return t, []

    with cf.ThreadPoolExecutor(8) as ex:
        for t, cs in ex.map(comments, tickets):
            acct = org2acct.get(t.get("organization_id"))
            obs.append(dict(system="zendesk", kind="ticket", ts=iso_ts(t.get("created_at")), author=t.get("requester_id"),
                            text=clean(f"{t.get('subject') or ''}. {t.get('description') or ''}"), account_id=acct,
                            thread=t["id"], channel="ticket", src_id=t["id"],
                            meta=dict(status=t.get("status"), priority=t.get("priority"), tags=t.get("tags"))))
            for c in cs[1:]:  # first comment repeats the description
                obs.append(dict(system="zendesk", kind="ticket_comment", ts=iso_ts(c.get("created_at")), author=c.get("author_id"),
                                text=clean(c.get("plain_body") or c.get("body") or ""), account_id=acct, thread=t["id"],
                                channel="ticket", src_id=f"{t['id']}:{c.get('id')}"))


# --------------------------------------------------------------------------------------------- gong
def export_gong(obs, gt):
    calls, cursor = [], None
    while True:
        params = dict(fromDateTime="2015-01-01T00:00:00Z", toDateTime="2030-01-01T00:00:00Z")
        if cursor:
            params["cursor"] = cursor
        res = get("gong", "/v2/calls", **params)
        calls += res.get("calls", [])
        cursor = res.get("records", {}).get("cursor")
        if not cursor:
            break
    ids = [c["id"] for c in calls]
    meta = {}
    for i in range(0, len(ids), 50):
        res = post("gong", "/v2/calls/extensive", {"filter": {"callIds": ids[i:i + 50]},
                                                   "contentSelector": {"context": "Extended", "exposedFields": {"parties": True}}})
        for c in res.get("calls", []):
            acct, deal = None, None
            for ctx in c.get("context", []) or []:
                for o in ctx.get("objects", []):
                    if o.get("objectType") == "Account":
                        acct = o.get("objectId")
                    if o.get("objectType") == "Opportunity":
                        deal = o.get("objectId")
            meta[c["metaData"]["id"]] = (c["metaData"], acct, deal)
    gt["calls"] = []
    for i in range(0, len(ids), 50):
        res = post("gong", "/v2/calls/transcript", {"filter": {"callIds": ids[i:i + 50]}})
        for tr in res.get("callTranscripts", []):
            md, acct, deal = meta.get(tr["callId"], ({}, None, None))
            gt["calls"].append(dict(id=tr["callId"], title=md.get("title"), account_id=acct, deal_id=deal,
                                    calendarEventId=md.get("calendarEventId"), started=md.get("started")))
            # one observation per (speaker turn group by topic): a call is many observations of one world state
            by_topic = {}
            for mono in tr.get("transcript", []):
                key = (mono.get("topic") or "call", mono.get("speakerId"))
                by_topic.setdefault(key, []).extend(s["text"] for s in mono.get("sentences", []))
            for (topic, spk), sents in by_topic.items():
                obs.append(dict(system="gong", kind="call_turns", ts=iso_ts(md.get("started")), author=spk,
                                text=clean(" ".join(sents)), account_id=acct, deal_id=deal, thread=tr["callId"],
                                channel=f"call:{topic}", src_id=f"{tr['callId']}:{topic}:{spk}"))


# --------------------------------------------------------------------------------------------- gmail
def _plain(payload):
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"] + "==").decode("utf-8", "replace")
    for p in payload.get("parts", []) or []:
        t = _plain(p)
        if t:
            return t
    return ""


def export_gmail(obs, gt, limit=None):
    ids, tok = [], None
    while True:
        params = dict(maxResults=500)
        if tok:
            params["pageToken"] = tok
        res = get("gmail", "/gmail/v1/users/me/messages", **params)
        ids += [m["id"] for m in res.get("messages", [])]
        tok = res.get("nextPageToken")
        if not tok or (limit and len(ids) >= limit):
            break
    ids = ids[:limit] if limit else ids

    def fetch(mid):
        try:
            return get("gmail", f"/gmail/v1/users/me/messages/{mid}", format="full")
        except Exception:
            return None

    with cf.ThreadPoolExecutor(12) as ex:
        for m in ex.map(fetch, ids):
            if not m:
                continue
            hd = {h["name"].lower(): h["value"] for h in m["payload"].get("headers", [])}
            body = _plain(m["payload"]) or m.get("snippet", "")
            body = body.split("\nOn ")[0]  # drop quoted history: it is another document's text
            obs.append(dict(system="gmail", kind="email", ts=int(m.get("internalDate", 0)) / 1000, author=hd.get("from"),
                            text=clean(f"{hd.get('subject', '')}. {body}"), account_id=hd.get("x-account-id"),
                            thread=m.get("threadId"), channel="email", src_id=m["id"], meta=dict(labels=m.get("labelIds"))))


# --------------------------------------------------------------------------------------------- calendar
def export_calendar(obs, gt, name2acct):
    items, tok = [], None
    while True:
        params = dict(maxResults=2500, singleEvents="true")
        if tok:
            params["pageToken"] = tok
        res = get("calendar", "/calendar/v3/calendars/primary/events", **params)
        items += res.get("items", [])
        tok = res.get("nextPageToken")
        if not tok:
            break
    call2acct = {c.get("calendarEventId"): c.get("account_id") for c in gt.get("calls", [])}
    for e in items:
        summ = e.get("summary") or ""
        acct = call2acct.get(e["id"]) or match_account(summ, name2acct)
        obs.append(dict(system="calendar", kind="event", ts=iso_ts((e.get("start") or {}).get("dateTime")),
                        author=(e.get("organizer") or {}).get("email"), text=clean(f"{summ}. {e.get('description') or ''}"),
                        account_id=acct, thread=e["id"], channel="calendar", src_id=e["id"]))


# --------------------------------------------------------------------------------------------- jira
def export_jira(obs, gt, name2acct):
    issues, start = [], 0
    while True:
        res = get("jira", "/rest/api/3/search", startAt=start, maxResults=100)
        issues += res.get("issues", [])
        start += len(res.get("issues", []))
        if not res.get("issues") or start >= res.get("total", 0):
            break
    for it in issues:
        f = it["fields"]
        cust = f.get("customfield_10053")
        acct = name2acct.get((cust or "").lower())
        obs.append(dict(system="jira", kind="issue", ts=iso_ts(f.get("created")), author=(f.get("reporter") or {}).get("accountId"),
                        text=clean(f"{f.get('summary') or ''}. {adf_text(f.get('description'))}"), account_id=acct,
                        thread=it["key"], channel=f"jira:{(f.get('project') or {}).get('key')}", src_id=it["key"],
                        meta=dict(status=(f.get("status") or {}).get("name"), priority=(f.get("priority") or {}).get("name"),
                                  labels=f.get("labels"), ticket=f.get("customfield_10054"))))


# --------------------------------------------------------------------------------------------- slack
def export_slack(obs, gt, name2acct):
    first = get("slack", "/_admin/records", limit=1)
    counts = first["counts"]
    gt["slack_users"] = get("slack", "/_admin/records", limit=10000)["users"]
    chans = {}
    for off in range(0, counts["channels"], 10000):
        for c in get("slack", "/_admin/records", limit=10000, offset=off)["channels"]:
            chans[c["id"]] = c
    gt["slack_channels"] = [{k: c.get(k) for k in ("id", "name", "is_private", "is_im", "is_mpim")} for c in chans.values()]
    n = max(counts["messages"], counts["replies"])
    for off in range(0, n, 10000):
        page = get("slack", "/_admin/records", limit=10000, offset=off)
        for kind in ("messages", "replies"):
            if off >= counts[kind]:
                continue
            for m in page[kind]:
                cid = m.get("channel_id") or ""
                cname = (chans.get(cid) or {}).get("name") or cid
                acct = cid if cid.startswith("acct-") else None
                if acct is None:  # e.g. ext-hatherly-payments, hatherly-payments-launch
                    acct = match_account(cname.replace("-", " "), name2acct)
                obs.append(dict(system="slack", kind="message" if kind == "messages" else "reply", ts=float(m["ts"]),
                                author=m.get("user"), text=clean(m.get("text")), account_id=acct,
                                thread=f"{cid}:{m.get('thread_ts') or m['ts']}", channel=cname, src_id=f"{cid}:{m['ts']}"))


# --------------------------------------------------------------------------------------------- helpers
def match_account(text, name2acct):
    t = (text or "").lower()
    best = None
    for name, acct in name2acct.items():
        if len(name) > 3 and name in t and (best is None or len(name) > len(best[0])):
            best = (name, acct)
    return best[1] if best else None


def export_truth(gt):
    for name, path in [("brief", "/_admin/tenants/default/brief"), ("people", "/_admin/tenants/default/people"),
                       ("dirt", "/_admin/tenants/default/dirt"), ("lookalikes", "/_admin/tenants/default/lookalikes")]:
        try:
            gt[name] = get("salesforce", path)
        except Exception as e:  # noqa: BLE001
            gt[name] = {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gmail-limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    obs, gt = [], {}
    t0 = time.time()
    export_salesforce(obs, gt); print(f"salesforce {len(obs)} {time.time()-t0:.0f}s", flush=True)
    name2acct = {a["Name"].lower(): a["Id"] for a in gt["accounts"]}
    export_zendesk(obs, gt); print(f"zendesk {len(obs)} {time.time()-t0:.0f}s", flush=True)
    export_gong(obs, gt); print(f"gong {len(obs)} {time.time()-t0:.0f}s", flush=True)
    export_calendar(obs, gt, name2acct); print(f"calendar {len(obs)} {time.time()-t0:.0f}s", flush=True)
    export_jira(obs, gt, name2acct); print(f"jira {len(obs)} {time.time()-t0:.0f}s", flush=True)
    export_slack(obs, gt, name2acct); print(f"slack {len(obs)} {time.time()-t0:.0f}s", flush=True)
    export_gmail(obs, gt, args.gmail_limit); print(f"gmail {len(obs)} {time.time()-t0:.0f}s", flush=True)
    export_truth(gt)
    for i, o in enumerate(obs):
        o["obs_id"] = i
        o.setdefault("deal_id", None)
        o.setdefault("meta", {})
    with open(os.path.join(args.out, "observations.jsonl"), "w") as f:
        for o in obs:
            f.write(json.dumps(o, default=str) + "\n")
    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(gt, f, default=str)
    print(f"done: {len(obs)} observations -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
