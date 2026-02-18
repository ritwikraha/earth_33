"""Heuristic (rule-based) agent for survival."""

from __future__ import annotations

from agents.base import BaseAgent
from config_io.schema import AgentAction, ActionType


class HeuristicAgent(BaseAgent):
    """Simple rule-based agent prioritizing survival needs."""

    def act(self, observation: dict) -> AgentAction:
        agent = observation["agent"]
        local = observation["local"]
        nearby = observation["nearby"]
        mask = set(observation.get("action_mask", []))

        hydration = agent["hydration"]
        energy = agent["energy"]
        fatigue = agent["fatigue"]
        core_temp = agent["core_temp_c"]
        injury = agent["injury"]
        water_avail = local["water_availability"]
        veg = local["vegetation_biomass"]
        air_temp = local["air_temp_c"]
        wildlife = local["wildlife_risk"]

        # Priority 1: Critical dehydration - drink if possible
        if hydration < 25 and water_avail > 0.1 and "DRINK" in mask:
            return AgentAction(
                action=ActionType.DRINK,
                reason="Critical dehydration, drinking available water",
                confidence=0.95,
            )

        # Priority 2: Critical energy - forage if possible
        if energy < 25 and veg > 0.1 and "FORAGE" in mask:
            return AgentAction(
                action=ActionType.FORAGE,
                reason="Critical energy, foraging available vegetation",
                confidence=0.90,
            )

        # Priority 3: Temperature management - shelter when cold or hot
        temp_dangerous = air_temp > 35 or air_temp < 5
        core_drifting = core_temp < 35.5 or core_temp > 38.5
        need_shelter = (temp_dangerous or core_drifting) and not agent.get("has_shelter", False)
        if need_shelter:
            # Use natural shelter if available
            if local["shelter_quality"] > 0.3:
                return AgentAction(
                    action=ActionType.REST,
                    reason=f"Temperature stress (air={air_temp:.0f}C, core={core_temp:.1f}C), using natural shelter",
                    confidence=0.90,
                )
            # Build shelter if we can
            if "BUILD_SHELTER" in mask and fatigue < 80:
                return AgentAction(
                    action=ActionType.BUILD_SHELTER,
                    reason="Temperature stress, building shelter",
                    confidence=0.85,
                )
            # Move toward best shelter
            best_shelter = nearby.get("best_shelter", {})
            if best_shelter.get("shelter_quality", 0) > 0.2:
                shelter_dir = best_shelter.get("direction", "")
                move_action = _dir_to_move(shelter_dir)
                if move_action and move_action.value in mask:
                    return AgentAction(
                        action=move_action,
                        reason=f"Seeking shelter ({shelter_dir})",
                        confidence=0.80,
                    )
            # If nothing else, rest to minimize exposure
            if core_drifting:
                return AgentAction(
                    action=ActionType.REST,
                    reason="Resting to conserve body heat",
                    confidence=0.75,
                )

        # Priority 4: High fatigue - rest
        if fatigue > 75:
            return AgentAction(
                action=ActionType.REST,
                reason="High fatigue, resting",
                confidence=0.85,
            )

        # Priority 5: Hunter avoidance (HIGH PRIORITY)
        visible_hunters = observation.get("visible_hunters", [])
        if visible_hunters:
            nearest_h = min(visible_hunters, key=lambda h: h["distance"])
            h_dist = nearest_h["distance"]
            hx = nearest_h["pos"]["x"]
            hy = nearest_h["pos"]["y"]
            ax = agent["pos"]["x"]
            ay = agent["pos"]["y"]

            if h_dist <= 6:
                # Danger zone: flee in opposite direction
                dx = ax - hx  # vector away from hunter
                dy = ay - hy
                if abs(dx) >= abs(dy):
                    flee_dir = "E" if dx > 0 else "W"
                else:
                    flee_dir = "S" if dy > 0 else "N"

                move_action = _dir_to_move(flee_dir)
                if move_action and move_action.value in mask:
                    return AgentAction(
                        action=move_action,
                        reason=f"Fleeing hunter at distance {h_dist}",
                        confidence=0.95,
                    )
                # If primary flee direction blocked, try perpendicular
                if flee_dir in ("E", "W"):
                    perp_dirs = ["N", "S"]
                else:
                    perp_dirs = ["E", "W"]
                for pd in perp_dirs:
                    move_action = _dir_to_move(pd)
                    if move_action and move_action.value in mask:
                        return AgentAction(
                            action=move_action,
                            reason=f"Evading hunter, moving {pd}",
                            confidence=0.85,
                        )

            elif h_dist <= 10:
                # Caution zone: hide if available
                if "HIDE" in mask:
                    return AgentAction(
                        action=ActionType.HIDE,
                        reason=f"Hunter spotted at distance {h_dist}, hiding",
                        confidence=0.80,
                    )

        # Priority 6: Wildlife danger - hide if risk is high
        if wildlife > 0.3 and "HIDE" in mask:
            return AgentAction(
                action=ActionType.HIDE,
                reason="High wildlife risk, hiding",
                confidence=0.75,
            )

        # Priority 7: Low hydration - move toward water
        if hydration < 50:
            water_dir = nearby.get("nearest_water", {}).get("direction", "")
            move_action = _dir_to_move(water_dir)
            if move_action and move_action.value in mask:
                return AgentAction(
                    action=move_action,
                    reason=f"Low hydration, moving toward water ({water_dir})",
                    confidence=0.70,
                )

        # Priority 8: Low energy - forage or move toward vegetation
        if energy < 50:
            if veg > 0.1 and "FORAGE" in mask:
                return AgentAction(
                    action=ActionType.FORAGE,
                    reason="Low energy, foraging",
                    confidence=0.70,
                )

        # Priority 9: Drink if water available and not full
        if hydration < 70 and water_avail > 0.2 and "DRINK" in mask:
            return AgentAction(
                action=ActionType.DRINK,
                reason="Topping up hydration",
                confidence=0.60,
            )

        # Priority 10: Forage if vegetation available and not full
        if energy < 70 and veg > 0.2 and "FORAGE" in mask:
            return AgentAction(
                action=ActionType.FORAGE,
                reason="Topping up energy",
                confidence=0.55,
            )

        # Priority 11: Moderate fatigue - rest
        if fatigue > 50:
            return AgentAction(
                action=ActionType.REST,
                reason="Moderate fatigue, resting",
                confidence=0.50,
            )

        # Priority 12: Trophy pursuit (follow hints)
        trophy_info = observation.get("trophy", {})
        if trophy_info:
            trophy_dir = trophy_info.get("trophy_direction")
            if trophy_dir:
                move_action = _dir_to_move(trophy_dir)
                if move_action and move_action.value in mask:
                    return AgentAction(
                        action=move_action,
                        reason=f"Moving toward trophy ({trophy_dir})",
                        confidence=0.50,
                    )

        # Default: move toward water if hydration not great, else explore
        if hydration < 60:
            water_dir = nearby.get("nearest_water", {}).get("direction", "")
            move_action = _dir_to_move(water_dir)
            if move_action and move_action.value in mask:
                return AgentAction(
                    action=move_action,
                    reason="Moving toward water supply",
                    confidence=0.45,
                )

        # Explore: move in any valid direction
        for act_name in ["MOVE_E", "MOVE_S", "MOVE_N", "MOVE_W"]:
            if act_name in mask:
                return AgentAction(
                    action=ActionType(act_name),
                    reason="Exploring",
                    confidence=0.30,
                )

        # Last resort: rest
        return AgentAction(
            action=ActionType.REST,
            reason="No better option, resting",
            confidence=0.20,
        )


def _dir_to_move(direction: str) -> ActionType | None:
    mapping = {
        "N": ActionType.MOVE_N,
        "S": ActionType.MOVE_S,
        "E": ActionType.MOVE_E,
        "W": ActionType.MOVE_W,
    }
    return mapping.get(direction)
