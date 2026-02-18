"""Tests for hunter NPC system."""

from __future__ import annotations

import numpy as np

from config_io.config import load_config, Config
from config_io.schema import Terrain, CauseOfDeath
from sim.world import WorldState
from sim.hunters import Hunter, HunterManager


def _make_config(enabled=True, count=5, min_r=3, max_r=8, margin=15) -> Config:
    """Create a config with hunters enabled."""
    config = load_config()
    config.hunters.enabled = enabled
    config.hunters.count = count
    config.hunters.min_detection_radius = min_r
    config.hunters.max_detection_radius = max_r
    config.hunters.spawn_margin = margin
    return config


def _make_world_and_manager(config: Config, seed: int = 42):
    """Create world and hunter manager."""
    world = WorldState(config, seed)
    sx, sy = world.find_spawn_point()
    rng = np.random.default_rng(seed + 1000)  # separate rng for hunters
    manager = HunterManager(config, world, rng, player_spawn=(sx, sy))
    return world, manager, sx, sy


def test_hunter_spawn_count():
    """Correct number of hunters spawn."""
    config = _make_config(count=5)
    _, manager, _, _ = _make_world_and_manager(config)
    assert len(manager.hunters) == 5


def test_hunter_spawn_distance():
    """All hunters spawn at least spawn_margin from player."""
    config = _make_config(count=5, margin=15)
    _, manager, sx, sy = _make_world_and_manager(config)
    for h in manager.hunters:
        dist = abs(h.x - sx) + abs(h.y - sy)
        assert dist >= 15, f"Hunter {h.id} at distance {dist} < 15"


def test_hunter_no_water_spawn():
    """Hunters never spawn on water."""
    config = _make_config(count=5)
    world, manager, _, _ = _make_world_and_manager(config)
    for h in manager.hunters:
        terrain = world.terrain_at(h.x, h.y)
        assert terrain != Terrain.WATER, f"Hunter {h.id} spawned on water"


def test_hunter_detection_radius_range():
    """All radii within configured min/max."""
    config = _make_config(count=10, min_r=3, max_r=8)
    _, manager, _, _ = _make_world_and_manager(config)
    for h in manager.hunters:
        assert 3 <= h.detection_radius <= 8, f"Radius {h.detection_radius} out of range"


def test_hunter_patrol_movement():
    """Patrolling hunter moves each step."""
    config = _make_config(count=1)
    _, manager, sx, sy = _make_world_and_manager(config)
    h = manager.hunters[0]
    old_x, old_y = h.x, h.y
    # Update away from the player (player is far)
    manager.update(0, 0)  # player at 0,0 which is far
    # Hunter should have moved at least once over many steps
    moved = False
    for _ in range(20):
        manager.update(0, 0)
        if h.x != old_x or h.y != old_y:
            moved = True
            break
    assert moved, "Hunter did not move during patrol"


def test_hunter_chase_activates():
    """Hunter starts chasing when player in radius."""
    config = _make_config(count=1, min_r=10, max_r=10)
    _, manager, _, _ = _make_world_and_manager(config)
    h = manager.hunters[0]
    # Place player right at hunter's position
    manager.update(h.x, h.y)
    assert h.is_chasing, "Hunter should be chasing when player is at same position"


def test_hunter_kill_on_contact():
    """Returns killer ID when hunter is adjacent to player."""
    config = _make_config(count=1, min_r=10, max_r=10)
    _, manager, _, _ = _make_world_and_manager(config)
    h = manager.hunters[0]
    # Place player right at hunter's position
    killer_id = manager.update(h.x, h.y)
    assert killer_id is not None, "Should have killed when at same position"
    assert killer_id == h.id


def test_hunter_no_kill_out_of_range():
    """Returns None when no hunter is adjacent."""
    config = _make_config(count=1, min_r=3, max_r=3, margin=5)
    _, manager, sx, sy = _make_world_and_manager(config)
    # Player at spawn which is far from hunters
    killer_id = manager.update(sx, sy)
    assert killer_id is None, "No hunter should be close enough to kill"


def test_hunter_disabled():
    """No hunters when enabled=False."""
    config = _make_config(enabled=False, count=5)
    _, manager, _, _ = _make_world_and_manager(config)
    assert len(manager.hunters) == 0


def test_visible_hunters_within_radius():
    """Only visible within observation radius."""
    config = _make_config(count=1, min_r=5, max_r=5)
    _, manager, _, _ = _make_world_and_manager(config)
    h = manager.hunters[0]
    # Query from far away
    visible = manager.get_visible_hunters(0, 0, visibility_radius=3)
    # If hunter is far from (0,0), should not be visible
    dist = abs(h.x) + abs(h.y)
    if dist > 3:
        assert len(visible) == 0
    # Query from hunter's position
    visible = manager.get_visible_hunters(h.x, h.y, visibility_radius=3)
    assert len(visible) == 1


def test_visible_hunters_no_radius_leak():
    """detection_radius not in visible data."""
    config = _make_config(count=1, min_r=5, max_r=5)
    _, manager, _, _ = _make_world_and_manager(config)
    h = manager.hunters[0]
    visible = manager.get_visible_hunters(h.x, h.y, visibility_radius=10)
    assert len(visible) == 1
    assert "detection_radius" not in visible[0], "detection_radius should not be leaked"
