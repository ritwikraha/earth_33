"""LLM agent with provider fallback chain: Gemini -> OpenAI -> heuristic."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from config_io.schema import AgentAction, ActionType, ACTION_JSON_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are controlling an organism trying to survive in a 2D Earth-like environment.

PRIMARY OBJECTIVE: Find and reach the TROPHY item on the map.
SURVIVAL OBJECTIVE: Stay alive by managing hydration, energy, body temperature, \
fatigue, injury, and infection.
THREAT: Hunter organisms patrol the map. If you enter their detection radius, they \
will chase and kill you. Each hunter has a DIFFERENT detection radius you must estimate.

PLANNING PROTOCOL (follow every turn):
1. ASSESS: Review vitals, position, and nearby threats
2. ANALYZE HUNTERS: For each visible hunter, estimate detection radius from:
   - Distance when first spotted (if NOT chasing, radius < that distance)
   - Chase activation distance (radius ~ that distance)
   - Movement patterns from sighting history
3. PLAN PATH: Consider 2-3 possible moves and their risks
4. TROPHY PURSUIT: Factor in trophy hints (warmer/colder, direction) when safe
5. DECIDE: Choose the safest action that makes progress toward the trophy

HUNTER RADIUS ESTIMATION:
- See hunter at distance D, NOT chasing -> radius is likely < D
- Hunter STARTS chasing at distance D -> radius is approximately D
- Give hunters a safety margin of +2 cells beyond your estimate

RULES:
- Output ONLY valid JSON matching the schema below. No other text.
- Keep "reason" concise: include threat assessment + plan (under 30 words).

ACTION SCHEMA:
{schema}

VALID ACTIONS: {actions}

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""

# Provider fallback chain: try each in order
DEFAULT_PROVIDER_CHAIN = [
    {"provider": "gemini", "model": "gemini-2.0-flash", "env_key": "GEMINI_API_KEY"},
    {"provider": "openai", "model": "gpt-4o-mini", "env_key": "OPENAI_API_KEY"},
]


def _load_env() -> None:
    """Load .env file from project root if available."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


class LLMAgent(BaseAgent):
    """LLM agent with automatic provider fallback on quota exhaustion.

    Tries Gemini first. If Gemini quota is exhausted, falls back to OpenAI.
    If both fail, falls back to the heuristic agent.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        max_retries: int = 2,
        temperature: float = 0.7,
    ):
        self.temperature = temperature
        self.max_retries = max_retries
        self._heuristic_fallback = HeuristicAgent()

        _load_env()

        # Build the provider chain: requested provider first, then others
        self._providers: list[_ProviderBackend] = []
        self._active_provider: _ProviderBackend | None = None
        self._quota_exhausted_providers: set[str] = set()

        # Add the requested provider first
        self._providers.append(
            _ProviderBackend(provider, model, temperature)
        )
        # Add remaining providers from the default chain as fallbacks
        for entry in DEFAULT_PROVIDER_CHAIN:
            if entry["provider"] != provider and os.environ.get(entry["env_key"]):
                self._providers.append(
                    _ProviderBackend(entry["provider"], entry["model"], temperature)
                )

        # Stats
        self.call_count: int = 0
        self.fallback_count: int = 0
        self.provider_switches: int = 0

    def act(self, observation: dict) -> AgentAction:
        action_mask = observation.get("action_mask", [a.value for a in ActionType])
        system = SYSTEM_PROMPT.format(
            schema=json.dumps(ACTION_JSON_SCHEMA, indent=2),
            actions=", ".join(action_mask),
        )
        user_msg = json.dumps(observation, indent=2)

        # Try each provider in the chain
        for backend in self._providers:
            if backend.name in self._quota_exhausted_providers:
                continue

            for attempt in range(1 + self.max_retries):
                try:
                    raw = backend.call(system, user_msg)
                    action = self._parse_response(raw, action_mask)
                    self.call_count += 1
                    self._active_provider = backend
                    return action

                except _QuotaExhaustedError:
                    # This provider's quota is done — mark it and move to next
                    self._quota_exhausted_providers.add(backend.name)
                    self.provider_switches += 1
                    logger.warning(
                        f"{backend.name} quota exhausted, "
                        f"switching to next provider"
                    )
                    break  # break retry loop, try next provider

                except Exception as e:
                    logger.warning(
                        f"{backend.name} attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < self.max_retries:
                        user_msg = (
                            f"Your previous response was invalid: {e}\n"
                            f"Please output ONLY valid JSON matching the schema.\n"
                            f"Observation:\n{json.dumps(observation, indent=2)}"
                        )
                    # If all retries exhausted for this provider,
                    # fall through to next provider
            else:
                # All retries for this provider failed (non-quota errors)
                # Try next provider
                continue

        # All providers exhausted — heuristic fallback
        self.fallback_count += 1
        logger.warning("All LLM providers failed, using heuristic fallback")
        return self._heuristic_fallback.act(observation)

    def _parse_response(self, raw: str, action_mask: list[str]) -> AgentAction:
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines)

        data = json.loads(raw)
        action = AgentAction(**data)

        if action.action.value not in action_mask:
            raise ValueError(
                f"Action {action.action.value} not in valid actions: {action_mask}"
            )
        return action

    @property
    def active_provider_name(self) -> str:
        if self._active_provider:
            return f"{self._active_provider.name}/{self._active_provider.model}"
        return "none"


class _QuotaExhaustedError(Exception):
    """Raised when a provider's quota is fully exhausted."""
    pass


class _ProviderBackend:
    """Wraps a single LLM provider (Gemini, OpenAI, Anthropic)."""

    def __init__(self, provider: str, model: str, temperature: float):
        self.name = provider
        self.model = model
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.name == "gemini":
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise _QuotaExhaustedError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)

        elif self.name == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise _QuotaExhaustedError("OPENAI_API_KEY not set")
            self._client = openai.OpenAI(api_key=api_key)

        elif self.name == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise _QuotaExhaustedError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {self.name}")

        return self._client

    def call(self, system: str, user_msg: str) -> str:
        client = self._get_client()

        if self.name == "gemini":
            return self._call_gemini(client, system, user_msg)
        elif self.name == "openai":
            return self._call_openai(client, system, user_msg)
        elif self.name == "anthropic":
            return self._call_anthropic(client, system, user_msg)
        raise ValueError(f"Unknown provider: {self.name}")

    def _call_gemini(self, client: Any, system: str, user_msg: str) -> str:
        from google.genai import types

        # One quick retry for transient rate limits, but treat sustained 429 as quota exhaustion
        for rate_attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self.temperature,
                        max_output_tokens=300,
                        response_mime_type="application/json",
                    ),
                )
                text = None
                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            text = part.text
                            break
                if text is None:
                    text = response.text
                if text is None:
                    raise ValueError("Gemini returned no text content")
                return text.strip()

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if "PerDay" in err_str:
                        # Daily quota — no point retrying, switch provider
                        raise _QuotaExhaustedError(
                            f"Gemini daily quota exhausted"
                        )
                    if rate_attempt == 0:
                        # Per-minute limit — wait and retry once
                        wait = _parse_retry_delay(err_str)
                        logger.info(
                            f"Gemini per-minute limit, waiting {wait:.0f}s..."
                        )
                        time.sleep(wait)
                        continue
                    else:
                        raise _QuotaExhaustedError(
                            "Gemini rate limit persists after retry"
                        )
                raise

        raise _QuotaExhaustedError("Gemini rate limited")

    def _call_openai(self, client: Any, system: str, user_msg: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "insufficient_quota" in err_str:
                raise _QuotaExhaustedError(f"OpenAI quota exhausted: {e}")
            raise

    def _call_anthropic(self, client: Any, system: str, user_msg: str) -> str:
        try:
            resp = client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                temperature=self.temperature,
                max_tokens=200,
            )
            return resp.content[0].text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                raise _QuotaExhaustedError(f"Anthropic quota exhausted: {e}")
            raise


def _parse_retry_delay(err_str: str) -> float:
    """Extract retry delay from error message."""
    match = re.search(r"retry in (\d+\.?\d*)", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0
    return 15.0
