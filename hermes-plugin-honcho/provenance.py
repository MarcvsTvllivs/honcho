"""Provenance hygiene for Honcho's automatic memory paths."""

from __future__ import annotations

import functools
import re

_GENERIC_ACTORS = ("the user", "the agent", "the assistant", "the system")

_ACTION_RE = (
    r"[^.!?\n]{0,240}\b(?:"
    r"executed|ran|invoked|used|called|pushed|committed|edited|created|updated|"
    r"wrote|read|searched|inspected|launched|started|completed|confirmed|"
    r"noted|notes"
    r")\b[^.!?\n]{0,240}\b(?:"
    r"background|command|terminal|tool|file|exit\s+code|git|process|task|"
    r"document|repository|repo|path|script|AGENTS\.md|\.md|shell|test"
    r")\b"
)


@functools.lru_cache(maxsize=16)
def _attribution_re(names: tuple[str, ...]) -> re.Pattern[str]:
    """Actor + operational verb + operational target, for the generic actors plus
    the configured peer names (Honcho writes its summaries with peer names)."""
    actors = "|".join(re.escape(a) for a in (*_GENERIC_ACTORS, *names))
    return re.compile(rf"\b(?:{actors})\b{_ACTION_RE}", re.IGNORECASE)


def _peer_names(user: str | None, agent: str | None) -> tuple[str, ...]:
    return tuple(sorted({n.strip() for n in (user, agent) if n and n.strip()}))


def provenance_guidance(user: str | None = None, agent: str | None = None) -> str:
    """Dialectic guardrail, naming the configured peers when known."""
    u = (user or "").strip() or "the user"
    a = (agent or "").strip() or "the agent"
    return (
        f"Provenance rules: {u} is the human user and {a} is the AI agent. "
        "Treat commands, tool calls, file edits, test runs, background jobs, and "
        "operational checks described in agent messages as actions performed by "
        f"the agent or system, not by {u}. Do not attribute those actions to "
        f"{u} unless a user message explicitly states that {u} personally "
        "performed the action. Distinguish commissioning a task from executing it. "
        "Prefer durable preferences, decisions, and constraints; omit routine "
        "execution details. When source is uncertain, use neutral wording such as "
        "'the agent' or 'the system'."
    )


def _sanitize_auto_injected_text(
    text: str, user: str | None = None, agent: str | None = None
) -> str:
    """Remove unsafe operational attributions from automatic Honcho recall.

    This is a conservative second line of defence. Source-peer labelling and
    Honcho deriver instructions remain the primary provenance controls. We drop
    only sentences that contain both an actor and an operational target, so a
    genuine preference ("<user> prefers concise responses") survives; we do not
    rewrite uncertain text into a claim we invented.
    """
    if not text:
        return ""

    pattern = _attribution_re(_peer_names(user, agent))
    cleaned_blocks: list[str] = []
    for block in re.split(r"\n{2,}", str(text)):
        sentences = re.split(r"(?<=[.!?])\s+", block.strip())
        kept = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip() and not pattern.search(sentence)
        ]
        if kept:
            cleaned_blocks.append(" ".join(kept))
    return "\n\n".join(cleaned_blocks).strip()


def config_peers(cfg) -> tuple[str | None, str | None]:
    """(user peer name, AI peer name) from a HonchoClientConfig, or (None, None)."""
    if cfg is None:
        return None, None
    return getattr(cfg, "peer_name", None), getattr(cfg, "ai_peer", None)
