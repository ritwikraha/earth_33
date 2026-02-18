"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from config_io.schema import AgentAction


class BaseAgent(ABC):
    """All agents must implement act()."""

    @abstractmethod
    def act(self, observation: dict) -> AgentAction:
        """Given an observation, return an action."""
        ...

    def reset(self) -> None:
        """Optional: reset agent state between episodes."""
        pass
