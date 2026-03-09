"""CLI command: replay a saved episode."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_io.config import Config
from config_io.utils import load_json
from config_io.schema import Terrain, CauseOfDeath
from sim.world import WorldState
from sim.physiology import OrganismState
from render.pygame_renderer import PygameRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved Earth33 episode")
    parser.add_argument("--replay", type=str, required=True, help="Path to replay JSON")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data = load_json(args.replay)
    config = Config(**data["config"])
    seed = data["meta"]["seed"]

    # Rebuild world (deterministic from seed)
    world = WorldState(config, seed)

    # Create renderer
    renderer = PygameRenderer(config, world)

    steps = data["steps"]
    base_fps = config.render.fps
    adjusted_delay = 1.0 / (base_fps * args.speed)

    logging.info(f"Replaying {len(steps)} steps (speed: {args.speed}x)")

    try:
        for step_data in steps:
            if not renderer.handle_events():
                break

            # Reconstruct organism state from logged data
            agent_data = step_data["agent"]
            pos = agent_data["pos"]
            org = OrganismState(
                x=pos["x"], y=pos["y"],
                hydration=agent_data["hydration"],
                energy=agent_data["energy"],
                core_temp_c=agent_data["core_temp_c"],
                fatigue=agent_data["fatigue"],
                injury=agent_data["injury"],
                infection=agent_data["infection"],
                alive=agent_data["alive"],
                cause_of_death=CauseOfDeath(agent_data.get("cause_of_death", "ALIVE")),
                has_shelter=agent_data.get("has_shelter", False),
            )

            action_str = step_data.get("action", {}).get("action", "")
            events = step_data.get("events", {})
            event_msgs = events.get("hazard_events", [])
            if events.get("encounter"):
                event_msgs.append("Wildlife encounter!")
            if events.get("rain"):
                event_msgs.append("Rain")

            time_info = step_data.get("time", {"day": 0, "hour": 0})
            # Rename for renderer
            time_for_render = {
                "day_of_year": time_info.get("day", 0),
                "hour": time_info.get("hour", 0),
            }

            renderer.render(
                org, step_data["t"], time_for_render,
                action_str, event_msgs,
            )

            time.sleep(adjusted_delay)

        # End-of-episode screen
        summary = data.get("summary", {})
        trophy_found = summary.get("trophy_found", False)
        logging.info(f"Replay complete. Survived: {summary.get('survived_hours', '?')} hours, "
                     f"Cause: {summary.get('cause_of_death', '?')}")

        if trophy_found and steps:
            # Reconstruct final state for victory screen
            last_step = steps[-1]
            agent_data = last_step["agent"]
            pos = agent_data["pos"]
            final_org = OrganismState(
                x=pos["x"], y=pos["y"],
                hydration=agent_data["hydration"],
                energy=agent_data["energy"],
                core_temp_c=agent_data["core_temp_c"],
                fatigue=agent_data["fatigue"],
                injury=agent_data["injury"],
                infection=agent_data["infection"],
                alive=agent_data["alive"],
                cause_of_death=CauseOfDeath(agent_data.get("cause_of_death", "ALIVE")),
                has_shelter=agent_data.get("has_shelter", False),
            )
            final_time = last_step.get("time", {"day": 0, "hour": 0})
            final_time_render = {
                "day_of_year": final_time.get("day", 0),
                "hour": final_time.get("hour", 0),
            }
            final_step_num = last_step["t"]

            # Get trophy position from replay events
            trophy_pos = None
            last_events_data = last_step.get("events", {})
            tp = last_events_data.get("trophy_pos")
            if tp:
                trophy_pos = (tp["x"], tp["y"])
            else:
                # Fallback: trophy is at the agent's final position (they found it)
                trophy_pos = (pos["x"], pos["y"])

            logging.info("Trophy found! Press ESC or close window.")

            # Victory animation (~4 seconds)
            anim_frames = base_fps * 4
            for f in range(anim_frames):
                if not renderer.handle_events():
                    break
                renderer.render_victory(
                    final_org, final_step_num, final_time_render,
                    trophy_pos, frame_offset=f,
                )
            else:
                # Hold victory screen
                while True:
                    if not renderer.handle_events():
                        break
                    renderer.render_victory(
                        final_org, final_step_num, final_time_render,
                        trophy_pos, frame_offset=anim_frames,
                    )
        else:
            logging.info("Press ESC or close window.")
            while True:
                if not renderer.handle_events():
                    break
                time.sleep(0.05)

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
