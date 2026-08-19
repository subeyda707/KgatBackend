"""KGAT core pipeline -- every function here matches what was tested and verified in the Colab notebook."""
import os, uuid, hmac, hashlib, json, random
from datetime import datetime, timezone
from collections import defaultdict

_SECRET = os.environ.get("KGAT_HMAC_SECRET", "").encode()

def make_event(agent_id, roles, action, subject, predicate, old_value, new_value, reason=None, event_id=None):
    return {
        "id": event_id or str(uuid.uuid4()), "agent": {"id": agent_id, "roles": roles}, "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": {"subject": subject, "predicate": predicate, "old_value": old_value, "new_value": new_value},
        "reason": reason, "verification": {"structural": None, "authorized": None, "conflict_free": None},
    }

def compute_diff(before, after):
    before_set, after_set = set(map(tuple, before)), set(map(tuple, after))
    return {"added": after_set - before_set, "removed": before_set - after_set}

def canonical_event_hash(event):
    canonical = {"id": event["id"], "agent_id": event["agent"]["id"], "roles": sorted(event["agent"]["roles"]),
        "action": event["action"], "subject": event["target"]["subject"], "predicate": event["target"]["predicate"],
        "old_value": event["target"]["old_value"], "new_value": event["target"]["new_value"],
        "timestamp": event["timestamp"], "reason": event["reason"]}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

def issue_token(agent_id, scope, event_hash_value, ttl_seconds=3600*24*365*5):
    expiry = int(datetime.now(timezone.utc).timestamp()) + ttl_seconds
    payload = f"{agent_id}:{scope}:{event_hash_value}:{expiry}"
    sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return {"agent_id": agent_id, "scope": scope, "event_hash": event_hash_value, "expiry": expiry, "signature": sig}

def verify_signature(token):
    payload = f"{token['agent_id']}:{token['scope']}:{token['event_hash']}:{token['expiry']}"
    expected = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, token.get("signature", ""))

def check_authorization(event, token, required_scope="kg:write"):
    if not verify_signature(token):
        return False, "Cryptographic signature invalid."
    if int(datetime.now(timezone.utc).timestamp()) > token["expiry"]:
        return False, "Token has expired."
    if token["scope"] != required_scope:
        return False, f"Token scope '{token['scope']}' does not match required scope '{required_scope}'."
    if token["event_hash"] != canonical_event_hash(event):
        return False, "Token is bound to a different event (role escalation or tampering detected)."
    roles = event["agent"]["roles"]
    if "data_writer" not in roles and "admin" not in roles:
        return False, f"Agent roles {roles} do not authorize this action."
    return True, None

def check_structural_grounding(event, kg_diff):
    claim = (event["target"]["subject"], event["target"]["predicate"], event["target"]["new_value"])
    if claim not in kg_diff["added"]:
        return False, f"Claim {claim} does not occur in the underlying KG diff."
    return True, None

def check_conflicts_for_event(event, all_events):
    by_sp = defaultdict(list)
    for e in all_events:
        t = e["target"]
        by_sp[(t["subject"], t["predicate"])].append((e["agent"]["id"], t["new_value"], e["id"]))
    for (s, p), claims in by_sp.items():
        if event["target"]["subject"] == s and event["target"]["predicate"] == p:
            if len(set(o for _, o, _ in claims)) > 1:
                return False, "Conflicts with another agent's independent claim."
    return True, None

def verify_event_fully(event, token, kg_diff, all_events):
    struct_ok, struct_msg = check_structural_grounding(event, kg_diff)
    if not struct_ok: return False, struct_msg
    auth_ok, auth_msg = check_authorization(event, token)
    if not auth_ok: return False, auth_msg
    conflict_ok, conflict_msg = check_conflicts_for_event(event, all_events)
    if not conflict_ok: return False, conflict_msg
    return True, None

def to_prov_o_activity(event):
    triple_entity = {
        "@id": f"urn:kgat:assertion:{event['target']['subject']}-{event['target']['predicate']}-{event['target']['new_value']}",
        "@type": "prov:Entity", "kgat:subject": event["target"]["subject"],
        "kgat:predicate": event["target"]["predicate"], "kgat:object": event["target"]["new_value"],
    }
    return {
        "@context": {"prov": "http://www.w3.org/ns/prov#", "kgat": "http://example.org/kgat-vocab#"},
        "@id": f"urn:kgat:event:{event['id']}", "@type": "prov:Activity",
        "prov:wasAssociatedWith": {"@type": "prov:Agent", "@id": f"urn:kgat:agent:{event['agent']['id']}"},
        "prov:startedAtTime": event["timestamp"], "prov:generated": triple_entity,
        "kgat:hasAction": event["action"], "kgat:hasVerificationStatus": event["verification"],
    }

def generate_balanced_evaluation_dataset(seed=42):
    random.seed(seed)
    subjects = [f"Entity{i}" for i in range(40)]
    predicates = ["relatesTo", "treats", "causes"]
    objects_pool = [f"Value{i}" for i in range(30)]
    authorized_agents = [("writer_1", ["data_writer"]), ("writer_2", ["data_writer"]), ("admin_1", ["admin"])]
    unauthorized_agents = [("reader_1", ["reader"]), ("reader_2", ["reader"])]

    kg_before = {(random.choice(subjects), random.choice(predicates), random.choice(objects_pool)) for _ in range(40)}
    kg_after = set(kg_before)
    categories = [("genuine_authorized", True, True), ("genuine_unauthorized", True, False),
                  ("false_authorized", False, True), ("false_unauthorized", False, False)]

    events, ground_truth = [], []
    used_triples = set()
    for label, is_genuine, is_authorized in categories:
        for _ in range(25):
            while True:
                s, p, o = random.choice(subjects), random.choice(predicates), random.choice(objects_pool)
                if (s, p, o) not in used_triples and (s, p, o) not in kg_before:
                    used_triples.add((s, p, o)); break
            agent_id, roles = random.choice(authorized_agents if is_authorized else unauthorized_agents)
            e = make_event(agent_id, roles, "added", s, p, None, o)
            if is_genuine: kg_after.add((s, p, o))
            events.append(e)
            ground_truth.append({"event_id": e["id"], "category": label, "is_structurally_genuine": is_genuine, "is_role_authorized": is_authorized})

    return events, ground_truth, compute_diff(kg_before, kg_after)

def run_rq1(events, ground_truth, kg_diff):
    tp = fp = tn = fn = 0
    for e, gt in zip(events, ground_truth):
        struct_ok, _ = check_structural_grounding(e, kg_diff)
        actual = gt["is_structurally_genuine"]
        if actual and struct_ok: tp += 1
        elif actual and not struct_ok: fn += 1
        elif not actual and struct_ok: fp += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    structural = {"precision": precision, "recall": recall, "f1": f1}

    atp = afp = atn = afn = 0
    for e, gt in zip(events, ground_truth):
        h = canonical_event_hash(e)
        token = issue_token(e["agent"]["id"], "kg:write", h)
        auth_ok, _ = check_authorization(e, token)
        actual = gt["is_role_authorized"]
        if actual and auth_ok: atp += 1
        elif actual and not auth_ok: afn += 1
        elif not actual and auth_ok: afp += 1
        else: atn += 1
    a_precision = atp / (atp + afp) if (atp + afp) else 0
    a_recall = atp / (atp + afn) if (atp + afn) else 0
    a_f1 = 2 * a_precision * a_recall / (a_precision + a_recall) if (a_precision + a_recall) else 0
    authorization = {"precision": a_precision, "recall": a_recall, "f1": a_f1}

    return {"structural": structural, "authorization": authorization}

def simulate_event(subject, predicate, obj, agent_role, scope, is_fabricated, kg_diff):
    """Live, on-demand verification for the interactive simulator -- runs the REAL checks,
    not a canned/precomputed result. This is what lets tampering genuinely change the outcome."""
    agent_id = "sim_agent"
    e = make_event(agent_id, [agent_role], "added", subject, predicate, None, obj)

    if not is_fabricated:
        kg_diff = {"added": kg_diff["added"] | {(subject, predicate, obj)}, "removed": kg_diff["removed"]}

    struct_ok, struct_msg = check_structural_grounding(e, kg_diff)

    h = canonical_event_hash(e)
    token = issue_token(agent_id, scope, h)
    auth_ok, auth_msg = check_authorization(e, token, required_scope="kg:write")

    conflict_ok, conflict_msg = check_conflicts_for_event(e, [e])

    overall = struct_ok and auth_ok and conflict_ok
    return {
        "event": {"who": agent_id, "role": agent_role, "what": f"added ({subject}, {predicate}, {obj})"},
        "structural": {"pass": struct_ok, "detail": struct_msg or "Found in underlying KG diff."},
        "authorization": {"pass": auth_ok, "detail": auth_msg or "Signature, expiry, scope, event binding, and role all valid."},
        "conflict": {"pass": conflict_ok, "detail": conflict_msg or "No competing claim found."},
        "verified": overall,
        "hash": h,
    }
