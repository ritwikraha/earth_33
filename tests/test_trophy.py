"""Tests for trophy objective system."""

from __future__ import annotations

import numpy as np

from config_io.config import load_config, Config
from config_io.schema import Terrain
from sim.world import WorldState
from sim.trophy import TrophyManager


def _make_config(enabled=True, min_dist=25, hint_interval=10) -> Config:
    """Create a config with trophy enabled."""
    config = load_config()
    config.trophy.enabled = enabled
    config.trophy.min_distance_from_spawn = min_dist
    config.trophy.hint_interval = hint_interval
    config.trophy.warm_cold_enabled = True
    return config


def _make_world_and_trophy(config: Config, seed: int = 42):
    """Create world and trophy manager."""
    world = WorldState(config, seed)
    sx, sy = world.find_spawn_point()
    rng = np.random.default_rng(seed + 2000)
    manager = TrophyManager(config, world, rng, player_spawn=(sx, sy))
    return world, manager, sx, sy


def test_trophy_placement_distance():
    """Trophy placed at min distance from spawn."""
    config = _make_config(min_dist=20)
    _, manager, sx, sy = _make_world_and_trophy(config)
    dist = abs(manager.trophy_x - sx) + abs(manager.trophy_y - sy)
    assert dist >= 20, f"Trophy at distance {dist} < 20"


def test_trophy_not_on_water():
    """Trophy never on water."""
    config = _make_config()
    world, manager, _, _ = _make_world_and_trophy(config)
    terrain = world.terrain_at(manager.trophy_x, manager.trophy_y)
    assert terrain != Terrain.WATER, "Trophy placed on water"


def test_trophy_found_adjacent():
    """check_found returns True when within 1 cell."""
    config = _make_config()
    _, manager, _, _ = _make_world_and_trophy(config)
    # Place player adjacent to trophy
    result = manager.check_found(manager.trophy_x + 1, manager.trophy_y)
    assert result is True


def test_trophy_found_same_cell():
    """check_found returns True when at same cell."""
    config = _make_config()
    _, manager, _, _ = _make_world_and_trophy(config)
    result = manager.check_found(manager.trophy_x, manager.trophy_y)
    assert result is True


def test_trophy_not_found_far():
    """check_found returns False when far away."""
    config = _make_config()
    _, manager, sx, sy = _make_world_and_trophy(config)
    # Player at spawn, trophy is far away
    dist = abs(manager.trophy_x - sx) + abs(manager.trophy_y - sy)
    if dist > 2:
        result = manager.check_found(sx, sy)
        assert result is False


def test_hint_warmer_colder():
    """Correctly reports warmer/colder based on distance change."""
    config = _make_config()
    _, manager, sx, sy = _make_world_and_trophy(config)
    tx, ty = manager.trophy_x, manager.trophy_y

    # First hint (reference distance)
    hint1 = manager.get_hint(sx, sy, step=0)

    # Move closer to trophy
    dx = 1 if tx > sx else -1
    dy = 1 if ty > sy else -1
    hint2 = manager.get_hint(sx + dx * 3, sy + dy * 3, step=1)
    assert hint2.get("trophy_temperature") == "warmer", f"Expected warmer, got {hint2}"

    # Move further from trophy
    hint3 = manager.get_hint(sx - dx * 5, sy - dy * 5, step=2)
    assert hint3.get("trophy_temperature") == "colder", f"Expected colder, got {hint3}"


def test_hint_direction():
    """Correct cardinal direction hint on hint_interval steps."""
    config = _make_config(hint_interval=5)
    _, manager, sx, sy = _make_world_and_trophy(config)
    tx, ty = manager.trophy_x, manager.trophy_y

    # Step 0 should give direction (0 % 5 == 0)
    hint = manager.get_hint(sx, sy, step=0)
    assert "trophy_direction" in hint
    direction = hint["trophy_direction"]
    # Verify direction is correct
    dx = tx - sx
    dy = ty - sy
    if abs(dx) > abs(dy):
        expected = "E" if dx > 0 else "W"
    else:
        expected = "S" if dy > 0 else "N"
    assert direction == expected, f"Expected {expected}, got {direction}"


def test_trophy_disabled():
    """No trophy placed when enabled=False."""
    config = _make_config(enabled=False)
    _, manager, _, _ = _make_world_and_trophy(config)
    assert manager.trophy_x == -1
    assert manager.trophy_y == -1
    assert manager.check_found(0, 0) is False
    assert manager.get_hint(0, 0, 0) == {}


def test_trophy_found_only_once():
    """Trophy can only be found once."""
    config = _make_config()
    _, manager, _, _ = _make_world_and_trophy(config)
    tx, ty = manager.trophy_x, manager.trophy_y
    result1 = manager.check_found(tx, ty)
    assert result1 is True
    result2 = manager.check_found(tx, ty)
    assert result2 is False, "Trophy should not be found again"
