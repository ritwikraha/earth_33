"""Replay logging: record and serialize episode data."""

from __future__ import annotations

from typing import Any

from config_io.utils import save_json, load_json


class ReplayLogger:
    """Accumulates step data and produces a replay JSON."""

    def __init__(self, seed: int, config_dict: dict):
        self.data: dict[str, Any] = {
            "meta": {"version": "0.1", "seed": seed},
            "config": config_dict,
            "steps": [],
            "summary": {},
        }

    def log_step(
        self,
        step: int,
        time_info: dict,
        agent_state: dict,
        local_state: dict,
        action: dict,
        events: dict,
    ) -> None:
        self.data["steps"].append({
            "t": step,
            "time": time_info,
            "agent": agent_state,
            "local": local_state,
            "action": action,
            "events": events,
        })

    def set_summary(self, summary: dict) -> None:
        self.data["summary"] = summary

    def save(self, path: str) -> None:
        save_json(self.data, path)

    @staticmethod
    def load(path: str) -> dict:
        return load_json(path)
