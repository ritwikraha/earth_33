"""Test: action schema validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from config_io.schema import AgentAction, ActionType


def test_valid_action():
    a = AgentAction(action=ActionType.MOVE_N, reason="going north", confidence=0.8)
    assert a.action == ActionType.MOVE_N
    assert a.reason == "going north"
    assert a.confidence == 0.8


def test_action_from_string():
    a = AgentAction(action="MOVE_E")
    assert a.action == ActionType.MOVE_E


def test_action_defaults():
    a = AgentAction(action=ActionType.REST)
    assert a.reason == ""
    assert a.confidence == 0.5


def test_invalid_action():
    with pytest.raises(ValidationError):
        AgentAction(action="FLY_UP")


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        AgentAction(action=ActionType.REST, confidence=1.5)
    with pytest.raises(ValidationError):
        AgentAction(action=ActionType.REST, confidence=-0.1)


def test_action_mask_validation():
    """Ensure all ActionType values are valid strings."""
    for at in ActionType:
        a = AgentAction(action=at)
        assert a.action == at


if __name__ == "__main__":
    test_valid_action()
    test_action_from_string()
    test_action_defaults()
    print("PASS: basic schema tests")
    # The pytest-dependent tests need pytest to run
    print("Run with pytest for full coverage")
