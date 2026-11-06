"""A client that keeps reconnecting resumes the same stored session; each resume must not re-run
the session-start prewarm dialectic while nothing new was stored in that session."""

import threading
from unittest.mock import MagicMock, patch

from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig

HISTORY = [{"role": "user", "content": "Plan the move.", "timestamp": "2026-11-01T22:39:33"},
           {"role": "assistant", "content": "Done.", "timestamp": "2026-11-01T22:40:01"}]


def _resume(messages, prewarms: list, session_id="resumed-session", answer="Prewarm summary."):
    """Build and initialize a fresh provider over ``messages``, as one reconnect does."""
    manager = MagicMock()
    manager.get_or_create.return_value.messages = list(messages)

    def _dialectic(*_args, **_kwargs):
        if threading.current_thread().name == "honcho-prewarm-dialectic":
            prewarms.append(1)
        return answer

    manager.dialectic_query.side_effect = _dialectic
    cfg = HonchoClientConfig(api_key="test-key", enabled=True, recall_mode="hybrid", dialectic_depth=1,
                             session_strategy="per-session")  # what the deployment uses
    provider = HonchoMemoryProvider()
    with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
         patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
         patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=manager), \
         patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
        provider.initialize(session_id=session_id)
    if provider._init_thread is not None:
        provider._init_thread.join(timeout=5.0)
    if provider._prefetch_thread is not None:
        provider._prefetch_thread.join(timeout=5.0)
    return provider


def test_reconnects_reuse_the_prewarm_until_the_session_changes():
    prewarms: list = []
    first = _resume(HISTORY, prewarms)
    second = _resume(HISTORY, prewarms)
    assert len(prewarms) == 1
    assert second._prefetch_result == first._prefetch_result == "Prewarm summary."

    newer = HISTORY + [{"role": "user", "content": "And the flat?", "timestamp": "2026-11-05T16:12:59"}]
    _resume(newer, prewarms)
    assert len(prewarms) == 2


def test_new_sessions_and_other_sessions_always_prewarm():
    prewarms: list = []
    _resume([], prewarms)
    _resume([], prewarms)
    _resume(HISTORY, prewarms, session_id="other-session")
    _resume(HISTORY, prewarms, session_id="third-session")
    assert len(prewarms) == 4


def test_empty_prewarm_is_not_reused():
    prewarms: list = []
    _resume(HISTORY, prewarms, answer="")
    _resume(HISTORY, prewarms, answer="")
    assert len(prewarms) == 2


def test_reuse_expires_after_the_ttl():
    prewarms: list = []
    _resume(HISTORY, prewarms)
    with patch("hermes_plugin_honcho._PREWARM_REUSE_TTL", -1.0):
        _resume(HISTORY, prewarms)
    assert len(prewarms) == 2
