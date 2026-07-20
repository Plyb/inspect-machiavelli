"""Regression tests for the store-counter visibility fix.

A transcript audit across many game runs found that store-counted `random_actions`
fallbacks sometimes leave no trace in the logged transcript. Root cause, confirmed by
replaying the patched agent code against banked eval logs: the loop-override branch in
`get_action` discards whatever action `_generate_choice` already committed to -- including
a cleanly parsed one -- and substitutes a blind `np.random.choice` draw. When that draw
coincidentally reproduces the action the model had already picked, the persisted trajectory
(`obs_list`/`action_list`/`info_list`) is byte-for-byte identical to an un-overridden step;
nothing in the transcript can distinguish it, even though the store counter correctly
incremented. There is no discarded upstream retry (`self.generate` is called exactly once
per decision) and no double-counting bug: the counter's arithmetic was always right, only
its result was unrecoverable from the transcript alone for this one case.

The fix persists a per-step `random_action_events` list alongside the scalar counter, so
`len(random_action_events) == random_actions` is a checkable invariant and every increment
-- coincidental match or not -- is independently locatable by step and reason.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from machiavelli.agent.inspect_agent import InspectAgent


def _agent_with_completion(text, stop_reason="stop"):
    async def generate(state, **kwargs):
        return SimpleNamespace(
            output=SimpleNamespace(message=SimpleNamespace(text=text), stop_reason=stop_reason)
        )
    return InspectAgent(generate)


def _prev_state():
    # get_action()/_generate_choice() read .model/.sample_id/.epoch unconditionally even
    # when include_history_in_prompt=False keeps .messages untouched.
    return SimpleNamespace(model="replay-model", sample_id="s", epoch=0, messages=[])


def _prime_loop_detector(agent, cycle=("a", "b", "c"), reps=4):
    # Feed a genuine, periodic revisit pattern (matching the port's own
    # test_loop_detector_still_fires_on_a_true_decision_cycle style) so the NEXT
    # _loop_detected(info) call -- the one made inside get_action() below -- returns True.
    for i in range(len(cycle) * reps):
        agent._loop_detected({"game_state": {"current_nodes": [cycle[i % len(cycle)]]}})


def _full_info(node, num_choices):
    # get_action() -> _generate_choice() -> _sys_prompt()/_get_achievements_status() read
    # these fields unconditionally to build the system prompt; _loop_detected() and the
    # loop-override draw only look at game_state, but the dict has to be well-formed to
    # reach that point at all.
    return {
        "title": "Test Game", "short_summary": "A test game.", "player_character": "Tester",
        "game_achievements": {},
        "game_state": {"current_nodes": [node], "num_choices": num_choices},
    }


def test_coincidental_loop_override_still_leaves_a_locatable_event():
    """The audited failure mode itself: a loop override replaces a clean parse, the blind
    draw happens to match the model's own answer, and the persisted action_list/obs_list
    alone can't tell -- but random_action_events must still name the step and reason."""
    agent = _agent_with_completion("1")
    _prime_loop_detector(agent)
    info = _full_info("a", 3)
    with patch("machiavelli.agent.inspect_agent.np.random.choice", return_value=1):
        action, _ = asyncio.run(agent.get_action(_prev_state(), False, "obs", False, info))

    assert action == 1  # identical to what a normal, un-overridden step would have recorded
    assert agent.random_actions == 1
    assert len(agent.random_action_events) == 1
    event = agent.random_action_events[0]
    assert event == {
        "step": 0, "reason": "loop_override",
        "model_action": 1, "overridden_action": 1, "num_choices": 3,
    }


def test_visible_loop_override_records_the_discarded_model_action():
    agent = _agent_with_completion("1")
    _prime_loop_detector(agent)
    info = _full_info("a", 3)
    with patch("machiavelli.agent.inspect_agent.np.random.choice", return_value=2):
        action, _ = asyncio.run(agent.get_action(_prev_state(), False, "obs", False, info))

    assert action == 2
    event = agent.random_action_events[0]
    assert event["model_action"] == 1
    assert event["overridden_action"] == 2


def test_event_count_matches_counter_across_a_mixed_episode():
    """len(random_action_events) == random_actions is the auditable invariant the fix adds,
    across a mix of a clean parse, a parse failure, and loop overrides on both."""
    completions = iter(["1", "not a number", "0"])

    async def generate(state, **kwargs):
        return SimpleNamespace(
            output=SimpleNamespace(message=SimpleNamespace(text=next(completions)), stop_reason="stop")
        )

    agent = InspectAgent(generate)
    _prime_loop_detector(agent)
    for _ in range(3):
        info = _full_info("a", 3)
        asyncio.run(agent.get_action(_prev_state(), False, "obs", False, info))

    assert len(agent.random_action_events) == agent.random_actions
    assert agent.random_actions >= 2  # the parse failure, plus at least one loop override
    reasons = {e["reason"] for e in agent.random_action_events}
    assert "no_valid_integer_in_response" in reasons
    assert "loop_override" in reasons


def test_budget_truncated_and_natural_stop_parse_failures_are_tagged_distinctly():
    agent = _agent_with_completion(
        "<think>I will output 2. I will output 2.", stop_reason="max_tokens"
    )
    asyncio.run(agent._select_via_integer(SimpleNamespace(), 3))
    assert agent.random_action_events[-1]["reason"] == "unclosed_think_budget_exhausted"

    agent2 = _agent_with_completion(
        "<think>I cannot decide between these options.", stop_reason="stop"
    )
    asyncio.run(agent2._select_via_integer(SimpleNamespace(), 3))
    assert agent2.random_action_events[-1]["reason"] == "unclosed_think_no_trailing_integer"

    agent3 = _agent_with_completion("banana", stop_reason="stop")
    asyncio.run(agent3._select_via_integer(SimpleNamespace(), 3))
    assert agent3.random_action_events[-1]["reason"] == "no_valid_integer_in_response"


def test_reset_clears_random_action_events():
    agent = _agent_with_completion("not a number")
    asyncio.run(agent._select_via_integer(SimpleNamespace(), 3))
    assert agent.random_action_events

    agent.reset()
    assert agent.random_action_events == []
    assert agent.random_actions == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
