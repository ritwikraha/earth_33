"""Random agent: picks uniformly from valid actions."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from agents.base import BaseAgent
from config_io.schema import AgentAction, ActionType


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 42):
        self.rng: Generator = np.random.default_rng(seed)

    def act(self, observation: dict) -> AgentAction:
        mask = observation.get("action_mask", [a.value for a in ActionType])
        choice = self.rng.choice(mask)
        return AgentAction(action=ActionType(choice), reason="random", confidence=0.0)

    def reset(self) -> None:
        pass
