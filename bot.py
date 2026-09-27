"""
bot.py — the HTTP server the magicpin AI Challenge judge harness talks to.

Implements the 5 endpoints from `challenge-testing-brief.md` §2:
  POST /v1/context   — receive a (possibly incremental) context push
  POST /v1/tick      — periodic wake-up; bot may proactively send
  POST /v1/reply      — a merchant/customer replied; bot must respond
  GET  /v1/healthz
  GET  /v1/metadata
Plus the optional POST /v1/teardown mentioned in §11 (wipe state at test end).

All composition logic lives in composer.py / conversation_handlers.py; this
file is just the transport + state layer: an idempotent versioned context
store, per-merchant suppression/dedup bookkeeping, and per-conversation
state for multi-turn replies.

Run:
    pip install fastapi uvicorn
    ANTHROPIC_API_KEY=sk-... uvicorn bot:app --host 0.0.0.0 --port 8080
(No API key -> falls back to the deterministic heuristic composer; see
composer.py / llm_client.py.)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from composer import compose
from conversation_handlers import ConversationState, Lookup, respond

app = FastAPI(title="magicpin AI Challenge — challenger bot")
START = time.time()

# ---------------------------------------------------------------------------
# In-memory state (per §2.1 of the testing brief: "Storing in memory is
# fine; just don't restart between calls.")
# ---------------------------------------------------------------------------

# (scope, context_id) -> {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> ConversationState
conversations: dict[str, ConversationState] = {}

# suppression_key -> last-sent unix ts (per-trigger dedup, independent of
# which conversation carried it — a trigger should fire at most once per
# validity window regardless of retries).
sent_suppression: dict[str, float] = {}

# merchant_id -> list of message bodies already sent this test window
# (anti-repetition input to composer.compose(recent_bodies=...)).
sent_bodies_by_merchant: dict[str, list[str]] = {}

TICK_TIME_BUDGET_S = 20.0  # stay comfortably under the judge's 30s/call cap
MAX_ACTIONS_PER_TICK = 20  # per §5 of the testing brief


def _get(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _lookup() -> Lookup:
    return Lookup(
        get_category=lambda slug: _get("category", slug),
        get_merchant=lambda mid: _get("merchant", mid),
        get_customer=lambda cid: _get("customer", cid),
    )


# ---------------------------------------------------------------------------
# GET /v1/healthz, GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Rahul Kumar"),
        "team_members": [os.environ.get("TEAM_NAME", "Rahul Kumar")],
        "model": os.environ.get("LLM_MODEL") or (
            "heuristic-composer (no LLM key set)" if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
            else os.environ.get("LLM_PROVIDER", "anthropic")
        ),
        "approach": (
            "4-context composer dispatched by trigger kind, LLM-first (temperature=0) with a deterministic "
            "per-kind heuristic fallback that never fabricates numbers/offers/citations; always re-grounds a "
            "trigger's claim against the merchant/customer's real fields before composing; rule-based fast paths "
            "for auto-reply detection, explicit opt-out, hostility, and intent-transition in /v1/reply."
        ),
        "contact_email": os.environ.get("TEAM_EMAIL", "rahulkumar_23bt103@dtu.ac.in"),
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression.clear()
    sent_bodies_by_merchant.clear()
    return {"accepted": True}


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope '{body.scope}'"}
    if not body.context_id:
        return {"accepted": False, "reason": "invalid_scope", "details": "context_id is required"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        if cur["version"] == body.version:
            # Idempotent re-post of the same version -> no-op success, not a
            # conflict (§2.1: "Re-posting the same version is a no-op").
            return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
                    "stored_at": datetime.now(timezone.utc).isoformat()}
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _is_expired(trigger: dict, now_iso: str) -> bool:
    expires_at = trigger.get("expires_at")
    if not expires_at:
        return False
    try:
        exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        return now > exp
    except ValueError:
        return False


@app.post("/v1/tick")
async def tick(body: TickBody):
    started = time.monotonic()
    lookup = _lookup()

    # Resolve every candidate trigger, then keep at most one per merchant
    # this tick (§14 FAQ: "only one action per (merchant_id, conversation_id)
    # pair per tick"), preferring the highest-urgency trigger when a merchant
    # has more than one ready.
    candidates: list[dict] = []
    for trigger_id in body.available_triggers:
        trigger = _get("trigger", trigger_id)
        if not trigger:
            continue
        if _is_expired(trigger, body.now):
            continue
        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and suppression_key in sent_suppression:
            continue  # already sent this exact trigger instance; don't repeat it
        merchant_id = trigger.get("merchant_id")
        merchant = _get("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue
        category = _get("category", merchant.get("category_slug"))
        if not category:
            continue
        customer_id = trigger.get("customer_id")
        customer = _get("customer", customer_id) if customer_id else None
        candidates.append({"trigger": trigger, "merchant": merchant, "category": category, "customer": customer})

    best_per_merchant: dict[str, dict] = {}
    for cand in candidates:
        mid = cand["merchant"]["merchant_id"]
        if mid not in best_per_merchant or cand["trigger"].get("urgency", 0) > best_per_merchant[mid]["trigger"].get("urgency", 0):
            best_per_merchant[mid] = cand

    actions = []
    for cand in best_per_merchant.values():
        if time.monotonic() - started > TICK_TIME_BUDGET_S:
            break  # §14 FAQ: return what we have rather than risk a timeout
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        trigger, merchant, category, customer = cand["trigger"], cand["merchant"], cand["category"], cand["customer"]
        merchant_id = merchant["merchant_id"]
        recent = tuple(sent_bodies_by_merchant.get(merchant_id, []))
        is_first_touch = merchant_id not in sent_bodies_by_merchant

        try:
            result = compose(category, merchant, trigger, customer=customer, is_first_touch=is_first_touch, recent_bodies=recent)
        except Exception as e:  # never let a single bad trigger 500 the whole tick
            continue
        if not result.get("body"):
            continue

        conversation_id = f"conv_{merchant_id}_{trigger['id']}"
        conversations[conversation_id] = ConversationState(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer["customer_id"] if customer else None,
            trigger_id=trigger["id"],
            category_slug=merchant.get("category_slug"),
            turns=[{"from": "bot", "body": result["body"]}],
            sent_bodies=[result["body"]],
        )

        suppression_key = trigger.get("suppression_key", "")
        if suppression_key:
            sent_suppression[suppression_key] = time.time()
        sent_bodies_by_merchant.setdefault(merchant_id, []).append(result["body"])

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer["customer_id"] if customer else None,
            "send_as": result["send_as"],
            "trigger_id": trigger["id"],
            "template_name": f"vera_challenger_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", ""), trigger.get("kind", "")],
            "body": result["body"],
            "cta": result["cta"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = conversations.get(body.conversation_id)
    if state is None:
        # A reply for a conversation we never started via /v1/tick (e.g. a
        # direct smoke test). Create minimal state from the request itself
        # rather than erroring — the rules in conversation_handlers.py
        # (auto-reply / opt-out / hostility / intent detection) don't
        # require prior context to work correctly.
        state = ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id,
            customer_id=body.customer_id,
        )
        conversations[body.conversation_id] = state

    result = respond(state, body.message, body.turn_number, _lookup())

    if result.get("body") and state.merchant_id:
        sent_bodies_by_merchant.setdefault(state.merchant_id, []).append(result["body"])

    out = {"action": result["action"], "rationale": result.get("rationale", "")}
    if result["action"] == "send":
        out["body"] = result.get("body", "")
        out["cta"] = result.get("cta", "none")
    elif result["action"] == "wait":
        out["wait_seconds"] = result.get("wait_seconds", 1800)
    return out
