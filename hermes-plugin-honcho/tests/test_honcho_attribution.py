"""Regression tests for Honcho provenance and automatic-injection hygiene."""

from types import SimpleNamespace

from hermes_plugin_honcho import (
    HonchoMemoryProvider,
    _is_internal_gateway_turn,
    _sanitize_auto_injected_text,
)


def test_task_notifications_are_not_user_turns():
    assert _is_internal_gateway_turn(
        '<task-notification>Background command completed (exit code 0)</task-notification>'
    )


def test_humans_can_discuss_internal_marker_text_normally():
    assert not _is_internal_gateway_turn(
        "Please explain what a <task-notification> block means in the hook."
    )


def test_auto_injection_drops_agent_operation_attributed_to_user():
    text = (
        "The user prefers concise responses.\n\n"
        "The user executed a background command to push a resolved-mechanism document "
        "and confirmed that it completed successfully with exit code 0.\n\n"
        "The user is working on a database migration."
    )

    cleaned = _sanitize_auto_injected_text(text)

    assert "prefers concise responses" in cleaned
    assert "working on a database migration" in cleaned
    assert "executed a background command" not in cleaned
    assert "exit code 0" not in cleaned


def test_auto_injection_uses_configured_peer_names():
    text = (
        "Alex prefers concise responses. "
        "Alex executed a background command to update a file. "
        "Robo completed a test in the repository."
    )

    assert _sanitize_auto_injected_text(text) == text
    cleaned = _sanitize_auto_injected_text(text, "Alex", "Robo")
    assert cleaned == "Alex prefers concise responses."


def test_auto_injection_drops_agent_operation_without_guessing_a_rewrite():
    text = "The agent ran a terminal command and edited a temporary file."

    assert _sanitize_auto_injected_text(text) == ""


def test_first_turn_context_sanitizes_every_injected_section():
    provider = HonchoMemoryProvider()
    provider._config = SimpleNamespace(peer_name="Alex", ai_peer="Robo")

    context = provider._format_first_turn_context({
        "summary": "Alex prefers concise responses.",
        "representation": "Alex executed a background command to update a file.",
        "card": "Alex is working on a database migration.",
        "ai_representation": "The agent ran a terminal command.",
        "ai_card": "Robo completed a test in the repository.",
    })

    assert "prefers concise responses" in context
    assert "working on a database migration" in context
    assert "executed a background command" not in context
    assert "ran a terminal command" not in context
    assert "completed a test" not in context


def test_dialectic_prompt_contains_provenance_guardrail():
    provider = HonchoMemoryProvider()

    prompt = provider._build_dialectic_prompt(0, [], True)
    assert "Do not attribute" in prompt
    assert "the user is the human user and the agent is the AI agent" in prompt

    provider._config = SimpleNamespace(peer_name="Alex", ai_peer="Robo")
    prompt = provider._build_dialectic_prompt(0, [], True)
    assert "Alex is the human user and Robo is the AI agent" in prompt
