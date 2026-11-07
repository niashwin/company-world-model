"""Access control for an organizational world model.

Every record carries the set of people who may read it (its readers), derived from the system it lives in. A world
model respects access control when everything a reader receives (search results, state vectors, answers) is computed
only from records that reader may read, and from model weights trained only on such records. That property is
non-interference: changing or deleting any record a reader cannot see changes nothing that reader receives.

There are three ways a learned world model can leak, and each needs its own control:
  retrieval    showing a record the reader cannot open                      -> filter search by the reader's records
  aggregation  a state vector averaged over records the reader cannot open  -> compute states from readable records
  training     weights shaped by records the reader cannot open            -> use an encoder trained on readable records

The policy below is an example for Era's simulated company (Northwind Robotics, 12 employees). It is role-based:
Era exposes roles, channel privacy and direct-message membership, but our export did not keep channel membership,
email recipients or internal-note flags, so finer rules are left for a richer export.
"""
import re

# employee id -> teams (from Era's people and Slack directory)
TEAMS = {
    "user-0": {"leadership"},                 # CEO
    "user-1": {"leadership", "engineering"},  # Head of Engineering
    "user-2": {"product"},                    # Product Marketing Manager
    "user-3": {"support"},                    # Support Engineer
    "user-4": {"finance"},                    # Financial Analyst
    "user-5": {"people"},                     # Recruiter
    "user-6": {"success"},                    # Customer Success Manager
    "user-7": {"product"},                    # Product Manager
    "user-8": {"engineering"},                # Software Engineer
    "user-9": {"sales"},                      # Account Executive
    "user-10": {"engineering"},               # Senior Software Engineer
    "user-11": {"engineering"},               # Platform Engineer
}
EVERYONE = frozenset(TEAMS)

# system -> teams that may read its records; "everyone" for company-wide systems
POLICY = {
    "slack": "everyone",                                                  # public channels; see private rules below
    "jira": "everyone",                                                   # an open issue tracker
    "zendesk": {"support", "success", "engineering", "product", "leadership"},
    "salesforce": {"sales", "success", "finance", "leadership"},
    "gong": {"sales", "success", "product", "leadership"},
    "gmail": {"sales", "success", "support", "leadership"},               # plus the sender
    "calendar": {"leadership"},                                           # plus the organizer
}
PRIVATE_CHANNELS = {"leadership", "exec-staff", "chan-leadership", "chan-exec-staff"}

POLICY_TABLE = [  # for the paper
    ("Slack, public channels (including customer channels)", "everyone"),
    ("Slack, private channels (leadership, exec-staff)", "leadership"),
    ("Slack, direct and group messages", "their members"),
    ("Jira (issues)", "everyone"),
    ("Zendesk (support tickets)", "support, success, engineering, product, leadership"),
    ("Salesforce (CRM)", "sales, success, finance, leadership"),
    ("Gong (call transcripts)", "sales, success, product, leadership"),
    ("Gmail", "the sender, plus sales, success, support, leadership"),
    ("Google Calendar", "the organizer, plus leadership"),
]

READERS = {  # the people whose views we evaluate
    "ceo": "user-0",
    "account_executive": "user-9",
    "support_engineer": "user-3",
    "software_engineer": "user-8",
    "financial_analyst": "user-4",
}


def members(teams):
    return frozenset(u for u, t in TEAMS.items() if t & set(teams))


def email_index(gt):
    return {p["email"].lower(): p["id"] for p in gt["people"]["people"]}


def record_readers(o, emails):
    """The set of employees who may read observation `o`."""
    system = o["system"]
    if system == "slack":
        ch = str(o.get("channel") or "")
        if ch.startswith("D-") or ch.startswith("mpdm-"):
            return frozenset(re.findall(r"user-\d+", ch))
        if ch in PRIVATE_CHANNELS:
            return members({"leadership"})
        return EVERYONE
    rule = POLICY.get(system, {"leadership"})
    if rule == "everyone":
        return EVERYONE
    readers = set(members(rule))
    if system in ("gmail", "calendar"):
        m = re.search(r"[\w.+-]+@[\w.-]+", str(o.get("author") or ""))
        if m and m.group(0).lower() in emails:
            readers.add(emails[m.group(0).lower()])
    return frozenset(readers)


def can_read(readers, user):
    return user in readers
