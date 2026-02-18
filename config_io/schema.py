"""JSON schemas and Pydantic models for validation."""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class Terrain(str, enum.Enum):
    PLAINS = "PLAINS"
    FOREST = "FOREST"
    DESERT = "DESERT"
    TUNDRA = "TUNDRA"
    MOUNTAIN = "MOUNTAIN"
    WATER = "WATER"
    SWAMP = "SWAMP"


class Soil(str, enum.Enum):
    SANDY = "SANDY"
    LOAMY = "LOAMY"
    CLAY = "CLAY"
    ROCKY = "ROCKY"
    PEAT = "PEAT"


class VegetationType(str, enum.Enum):
    NONE = "NONE"
    GRASS = "GRASS"
    SHRUB = "SHRUB"
    TREES = "TREES"


class ActionType(str, enum.Enum):
    MOVE_N = "MOVE_N"
    MOVE_S = "MOVE_S"
    MOVE_E = "MOVE_E"
    MOVE_W = "MOVE_W"
    REST = "REST"
    DRINK = "DRINK"
    FORAGE = "FORAGE"
    BUILD_SHELTER = "BUILD_SHELTER"
    HIDE = "HIDE"
    SIGNAL = "SIGNAL"


class CauseOfDeath(str, enum.Enum):
    ALIVE = "ALIVE"
    DEHYDRATION = "DEHYDRATION"
    STARVATION = "STARVATION"
    HYPOTHERMIA = "HYPOTHERMIA"
    HYPERTHERMIA = "HYPERTHERMIA"
    TRAUMA = "TRAUMA"
    INFECTION = "INFECTION"
    HUNTED = "HUNTED"


class EpisodeOutcome(str, enum.Enum):
    RUNNING = "RUNNING"
    DIED = "DIED"
    TROPHY_FOUND = "TROPHY_FOUND"


# ── Action schema (what the LLM must produce) ─────────────────────────────

class AgentAction(BaseModel):
    action: ActionType
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ── JSON schema dict for the action (for prompt inclusion) ─────────────────

ACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [a.value for a in ActionType],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["action"],
    "additionalProperties": False,
}


# ── Direction helpers ──────────────────────────────────────────────────────

ACTION_DIRECTION: dict[ActionType, tuple[int, int]] = {
    ActionType.MOVE_N: (0, -1),
    ActionType.MOVE_S: (0, 1),
    ActionType.MOVE_E: (1, 0),
    ActionType.MOVE_W: (-1, 0),
}
