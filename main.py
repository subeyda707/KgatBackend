import os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google import genai

from kgat_pipeline import (
    generate_balanced_evaluation_dataset, run_rq1, verify_event_fully,
    issue_token, canonical_event_hash, to_prov_o_activity, simulate_event,
    check_structural_grounding,
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
MODEL_NAME = "gemini-3.6-flash"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

class EventDocumentation(BaseModel):
    event_id: str
    who: str
    what: str
    when: str
    why: Optional[str] = None

_cache = {}

def _build_pipeline_results():
    events, ground_truth, kg_diff = generate_balanced_evaluation_dataset()
    rq1 = run_rq1(events, ground_truth, kg_diff)

    outcomes = []
    for e in events:
        h = canonical_event_hash(e)
        token = issue_token(e["agent"]["id"], "kg:write", h)
        verified, reason = verify_event_fully(e, token, kg_diff, events)
        outcomes.append((e, token, verified, reason))

    audit_trail = [
        {
            "id": e["id"], "who": e["agent"]["id"], "roles": e["agent"]["roles"],
            "what": f"{e['action']} ({e['target']['subject']}, {e['target']['predicate']}, {e['target']['new_value']})",
            "status": "VERIFIED" if verified else "REJECTED",
            "reason": reason,
            "hash": canonical_event_hash(e),
            "prov_o": to_prov_o_activity(e) if verified else None,
        }
        for e, token, verified, reason in outcomes
    ]
    return {"audit_trail": audit_trail, "rq1": rq1}, outcomes, kg_diff, events

@app.on_event("startup")
def startup():
    results, outcomes, kg_diff, events = _build_pipeline_results()
    _cache["results"] = results
    _cache["outcomes"] = outcomes
    _cache["kg_diff"] = kg_diff
    _cache["events"] = events

@app.get("/api/results")
def get_results():
    return _cache["results"]

@app.post("/api/document")
def document_event(event_id: str):
    for e, token, verified, reason in _cache["outcomes"]:
        if e["id"] == event_id:
            if not verified:
                return {"error": f"Event rejected: {reason}"}
            if not client:
                return {"error": "GEMINI_API_KEY not configured on server."}
            prompt = f"""Produce structured documentation. State ONLY these fields, invent nothing:
WHO: {e['agent']['id']}
WHAT: {e['action']} ({e['target']['subject']}, {e['target']['predicate']}, {e['target']['new_value']})
WHEN: {e['timestamp']}
WHY: {e['reason'] or 'Not recorded'}"""
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": EventDocumentation.model_json_schema()})
            return json.loads(response.text)
    return {"error": "Event not found."}

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not client:
        return {"answer": "GEMINI_API_KEY not configured on server."}
    lines = ["You are answering questions about a Knowledge Graph audit trail.",
             "ONLY reference the records below. If asked about something not listed, say so plainly.", ""]
    for r in _cache["results"]["audit_trail"]:
        lines.append(f"- {r['who']}: {r['what']} -- status={r['status']}{' ('+r['reason']+')' if r['reason'] else ''}")
    prompt = "\n".join(lines) + f"\n\nQuestion: {req.question}"
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return {"answer": response.text}

@app.get("/api/health")
def health():
    return {"status": "ok", "gemini_configured": client is not None}

class SimulateRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    agent_role: str
    scope: str = "kg:write"
    is_fabricated: bool = False

@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Live verification for the interactive simulator -- real checks, real tampering effects."""
    kg_diff = _cache.get("kg_diff", {"added": set(), "removed": set()})
    result = simulate_event(req.subject, req.predicate, req.object, req.agent_role, req.scope, req.is_fabricated, kg_diff)

    if result["verified"] and client:
        prompt = f"""Produce structured documentation. State ONLY these fields, invent nothing:
WHO: {result['event']['who']}
WHAT: {result['event']['what']}
WHEN: now
WHY: Not recorded"""
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": EventDocumentation.model_json_schema()})
            result["documentation"] = json.loads(response.text)
        except Exception as e:
            result["documentation"] = None
            result["documentation_error"] = str(e)
    else:
        result["documentation"] = None

    return result

class CompareRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    is_fabricated: bool = False

@app.post("/api/compare")
def compare(req: CompareRequest):
    """Runs the SAME event through both conditions live -- this is the actual RQ2 experiment,
    experienced rather than just reported as a number."""
    kg_diff = _cache.get("kg_diff", {"added": set(), "removed": set()})
    if not req.is_fabricated:
        kg_diff = {"added": kg_diff["added"] | {(req.subject, req.predicate, req.object)}, "removed": kg_diff["removed"]}

    struct_ok, struct_msg = check_structural_grounding(
        {"target": {"subject": req.subject, "predicate": req.predicate, "new_value": req.object}}, kg_diff
    )

    result = {"structural_pass": struct_ok, "structural_detail": struct_msg}

    if not client:
        result["baseline"] = {"error": "GEMINI_API_KEY not configured on server."}
        result["constrained"] = {"error": "GEMINI_API_KEY not configured on server."}
        return result

    baseline_prompt = f"Explain why this Knowledge Graph change happened: {req.subject} now {req.predicate} {req.object}."
    try:
        baseline_response = client.models.generate_content(model=MODEL_NAME, contents=baseline_prompt)
        result["baseline"] = {"documented": True, "text": baseline_response.text}
    except Exception as e:
        result["baseline"] = {"documented": False, "error": str(e)}

    if struct_ok:
        constrained_prompt = f"""Produce structured documentation. State ONLY these fields, invent nothing:
WHO: sim_agent
WHAT: added ({req.subject}, {req.predicate}, {req.object})
WHEN: now
WHY: Not recorded"""
        try:
            constrained_response = client.models.generate_content(
                model=MODEL_NAME, contents=constrained_prompt,
                config={"response_mime_type": "application/json", "response_schema": EventDocumentation.model_json_schema()})
            result["constrained"] = {"documented": True, "json": json.loads(constrained_response.text)}
        except Exception as e:
            result["constrained"] = {"documented": False, "error": str(e)}
    else:
        result["constrained"] = {"documented": False, "reason": struct_msg}

    return result
