"""Regression tests for the thinking-register fixes.

Cases captured from 5-game validation runs with a thinking-register (<think>-emitting) model:
- fate-of-the-storm-gods decision 108: a converged, stop-terminated 86-token completion
  ending in a bare valid integer but missing the literal </think> close must parse, not
  fall to a random action.
- avatar-of-the-wolf decisions 3-11: a shared 5-node scene footer pushed per decision must
  NOT fill the loop-detector window when only one representative node is tracked per
  decision.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from machiavelli.agent.inspect_agent import InspectAgent


def _agent_with_completion(text, stop_reason):
    async def generate(state, **kwargs):
        return SimpleNamespace(
            output=SimpleNamespace(
                message=SimpleNamespace(text=text), stop_reason=stop_reason
            )
        )
    return InspectAgent(generate)


def _select(agent, num_actions=3):
    return asyncio.run(agent._select_via_integer(SimpleNamespace(), num_actions))


def test_converged_unclosed_think_with_natural_stop_parses_trailing_integer():
    agent = _agent_with_completion(
        "<think>The sanctum is collapsing... grabbing the cracked floor piece risks "
        "breaking under your weight.\n0",
        stop_reason="stop",
    )
    action, _ = _select(agent)
    assert action == 0
    assert agent.random_actions == 0


def test_budget_truncated_unclosed_think_still_falls_back():
    agent = _agent_with_completion(
        "<think>I will output 2. I will output 2. I will output 2.",
        stop_reason="max_tokens",
    )
    _select(agent)
    assert agent.random_actions == 1


def test_closed_think_path_unchanged():
    agent = _agent_with_completion("<think>reasoning</think>\n1", stop_reason="stop")
    action, _ = _select(agent)
    assert action == 1
    assert agent.random_actions == 0


def test_natural_stop_without_trailing_integer_still_falls_back():
    agent = _agent_with_completion(
        "<think>I cannot decide between these options.", stop_reason="stop"
    )
    _select(agent)
    assert agent.random_actions == 1


def test_loop_detector_ignores_shared_multi_node_scene_footer():
    # avatar-of-the-wolf canary: every decision pushes a distinct head node plus the same
    # 5-node footer. Per-decision representative tracking must not fire on the footer.
    agent = InspectAgent(generate=None)
    footer = ["1_profaned_110", "1_profaned_112", "1_profaned_113",
              "1_profaned_115_collapsed", "1_profaned_117"]
    fired = []
    for i in range(12):
        info = {"game_state": {"current_nodes": [f"head_{i}", *footer]}}
        fired.append(agent._loop_detected(info))
    assert not any(fired)


def test_loop_detector_still_fires_on_a_true_decision_cycle():
    agent = InspectAgent(generate=None)
    fired = []
    for i in range(18):
        node = f"cycle_{i % 6}"
        fired.append(agent._loop_detected({"game_state": {"current_nodes": [node]}}))
    assert any(fired)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
