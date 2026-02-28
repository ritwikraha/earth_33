"""HuggingFace Transformers-based LLM agent for open-source models.

Loads any HuggingFace causal LM (e.g., Qwen2.5, Llama-3, Mistral, Gemma, Phi)
with optional 4-bit quantization for running on Colab GPUs.

Supports:
  - Configurable system prompts and reasoning strategies
  - Observation memory window for multi-turn context
  - In-context learning with survival demonstrations
  - Temperature-controlled sampling
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.base import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from config_io.schema import AgentAction, ActionType, ACTION_JSON_SCHEMA

logger = logging.getLogger(__name__)

# ── System prompts for different reasoning strategies ────────────────────────

DIRECT_PROMPT = """\
You are controlling an organism trying to survive in a 2D Earth-like environment.

PRIMARY OBJECTIVE: Find and reach the TROPHY item on the map.
SURVIVAL OBJECTIVE: Stay alive by managing hydration, energy, body temperature, \
fatigue, injury, and infection.
THREAT: Hunter organisms patrol the map. If you enter their detection radius, they \
will chase and kill you.

RULES:
- Output ONLY valid JSON matching the schema below. No other text.

ACTION SCHEMA:
{schema}

VALID ACTIONS: {actions}

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""

COT_PROMPT = """\
You are controlling an organism trying to survive in a 2D Earth-like environment.

PRIMARY OBJECTIVE: Find and reach the TROPHY item on the map.
SURVIVAL OBJECTIVE: Stay alive by managing hydration, energy, body temperature, \
fatigue, injury, and infection.
THREAT: Hunter organisms patrol the map. If you enter their detection radius, they \
will chase and kill you.

Think step-by-step about the current situation before choosing your action:
1. What are my most critical vitals right now?
2. Are there any immediate threats?
3. What action best balances survival and progress?

Then output ONLY valid JSON matching the schema below. No other text after the JSON.

ACTION SCHEMA:
{schema}

VALID ACTIONS: {actions}

Output your brief reasoning, then the JSON object on its own line.
"""

STRUCTURED_PROMPT = """\
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

REASONING_PROMPTS = {
    "direct": DIRECT_PROMPT,
    "cot": COT_PROMPT,
    "structured": STRUCTURED_PROMPT,
}

# ── In-context learning demonstrations ──────────────────────────────────────

ICL_DEMONSTRATIONS = [
    {
        "observation_summary": (
            "Step 12. Hydration: 45%, Energy: 72%, Temp: 36.8C. "
            "Terrain: PLAINS. Water nearby to the East (dist 2). "
            "No hunters visible. Trophy hint: warmer, direction E."
        ),
        "action": {"action": "DRINK", "reason": "Hydration dropping below 50%, prioritize water before it becomes critical", "confidence": 0.9},
    },
    {
        "observation_summary": (
            "Step 38. Hydration: 78%, Energy: 22%, Temp: 37.1C. "
            "Terrain: FOREST. High vegetation biomass. "
            "Hunter 3 cells away, not chasing. Trophy hint: colder."
        ),
        "action": {"action": "FORAGE", "reason": "Energy critically low at 22%, forest has food. Hunter at safe distance", "confidence": 0.85},
    },
    {
        "observation_summary": (
            "Step 55. Hydration: 60%, Energy: 55%, Temp: 33.2C. "
            "Terrain: TUNDRA. Air temp: -8C. No shelter. "
            "No hunters visible. Trophy: moderate distance."
        ),
        "action": {"action": "BUILD_SHELTER", "reason": "Core temp 33.2C approaching hypothermia at 30C, must build shelter immediately", "confidence": 0.95},
    },
]


def _format_icl_examples(num_shots: int) -> str:
    """Format ICL demonstrations as a conversation prefix."""
    if num_shots <= 0:
        return ""

    examples = ICL_DEMONSTRATIONS[:num_shots]
    parts = ["\nHere are examples of good survival decisions:\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Situation: {ex['observation_summary']}")
        parts.append(f"Response: {json.dumps(ex['action'])}\n")
    parts.append("Now handle the current observation:\n")
    return "\n".join(parts)


class HuggingFaceAgent(BaseAgent):
    """Agent powered by a locally loaded HuggingFace Transformers model.

    Args:
        model_name: HuggingFace model identifier (e.g. 'Qwen/Qwen2.5-7B-Instruct').
        quantize_4bit: Use 4-bit quantization via bitsandbytes (recommended for >3B).
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate per turn.
        reasoning_strategy: One of 'direct', 'cot', 'structured'.
        memory_window: Number of past observation summaries to include (0=disabled).
        num_icl_shots: Number of in-context learning examples (0=disabled).
        custom_system_prompt: Override the default system prompt entirely.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quantize_4bit: bool = True,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        reasoning_strategy: str = "structured",
        memory_window: int = 0,
        num_icl_shots: int = 0,
        custom_system_prompt: str | None = None,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.reasoning_strategy = reasoning_strategy
        self.memory_window = memory_window
        self.num_icl_shots = num_icl_shots
        self._heuristic_fallback = HeuristicAgent()

        # Stats
        self.call_count: int = 0
        self.fallback_count: int = 0
        self.parse_errors: int = 0

        # Memory buffer
        self._memory: list[dict[str, Any]] = []

        # Select system prompt
        if custom_system_prompt:
            self._system_prompt_template = custom_system_prompt
        else:
            self._system_prompt_template = REASONING_PROMPTS.get(
                reasoning_strategy, STRUCTURED_PROMPT
            )

        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        logger.info(f"Loading model: {model_name} (4bit={quantize_4bit})")
        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        self.model.eval()
        logger.info(f"Model loaded: {model_name}")

    def act(self, observation: dict) -> AgentAction:
        import torch

        action_mask = observation.get(
            "action_mask", [a.value for a in ActionType]
        )

        # Build system prompt
        system = self._system_prompt_template.format(
            schema=json.dumps(ACTION_JSON_SCHEMA, indent=2),
            actions=", ".join(action_mask),
        )

        # Add ICL examples if configured
        icl_text = _format_icl_examples(self.num_icl_shots)

        # Build memory context
        memory_text = ""
        if self.memory_window > 0 and self._memory:
            window = self._memory[-self.memory_window:]
            lines = ["Recent history:"]
            for m in window:
                lines.append(
                    f"  Step {m['step']}: {m['action']} | "
                    f"H={m['hydration']:.0f}% E={m['energy']:.0f}% "
                    f"T={m['temp']:.1f}C"
                )
            memory_text = "\n".join(lines) + "\n\n"

        # Build user message
        user_msg = memory_text + icl_text + json.dumps(observation, indent=2)

        # Build chat messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        # Tokenize
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without chat template
            prompt = f"System: {system}\n\nUser: {user_msg}\n\nAssistant:"
            input_ids = self.tokenizer.encode(
                prompt, return_tensors="pt"
            )

        input_ids = input_ids.to(self.model.device)

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=max(self.temperature, 0.01),
                    do_sample=self.temperature > 0.01,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            action = self._parse_response(response, action_mask)
            self.call_count += 1

            # Record to memory
            if self.memory_window > 0:
                agent_data = observation.get("agent", {})
                self._memory.append({
                    "step": observation.get("timestep", 0),
                    "action": action.action.value,
                    "hydration": agent_data.get("hydration", 0),
                    "energy": agent_data.get("energy", 0),
                    "temp": agent_data.get("core_temp_c", 37),
                })

            return action

        except Exception as e:
            logger.warning(f"HuggingFace generation failed: {e}")
            self.fallback_count += 1
            return self._heuristic_fallback.act(observation)

    def _parse_response(self, raw: str, action_mask: list[str]) -> AgentAction:
        """Extract JSON action from model response, handling various formats."""
        # Strip markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Try direct JSON parse
        try:
            data = json.loads(cleaned)
            return self._validate_action(data, action_mask)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to extract JSON object from text
        json_match = re.search(r'\{[^{}]*"action"\s*:\s*"[A-Z_]+"[^{}]*\}', cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_action(data, action_mask)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find any valid action string in the response
        for action_name in action_mask:
            if action_name in cleaned.upper():
                self.parse_errors += 1
                return AgentAction(
                    action=ActionType(action_name),
                    reason="parsed from free text",
                    confidence=0.3,
                )

        # Complete failure - use heuristic
        self.parse_errors += 1
        raise ValueError(f"Could not parse action from response: {cleaned[:200]}")

    def _validate_action(
        self, data: dict, action_mask: list[str]
    ) -> AgentAction:
        action = AgentAction(**data)
        if action.action.value not in action_mask:
            raise ValueError(
                f"Action {action.action.value} not in valid actions: {action_mask}"
            )
        return action

    def reset(self) -> None:
        self._memory.clear()
        self.call_count = 0
        self.fallback_count = 0
        self.parse_errors = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "fallbacks": self.fallback_count,
            "parse_errors": self.parse_errors,
            "memory_size": len(self._memory),
        }
