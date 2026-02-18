"""Pygame renderer: terrain layers, overlays, agent, hunters, trophy, fog, HUD."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from config_io.config import Config
from config_io.schema import Terrain
from render.palettes import (
    TERRAIN_COLORS_BY_IDX,
    temperature_color, water_color, vegetation_color,
    wildlife_color, shelter_color, elevation_color,
    HUD_BG, HUD_TEXT, HUD_BAR_BG,
    BAR_HYDRATION, BAR_ENERGY, BAR_FATIGUE, BAR_INJURY, BAR_INFECTION,
    AGENT_COLOR, TRAIL_COLOR,
    HUNTER_COLOR, HUNTER_CHASING_COLOR, HUNTER_RADIUS_ALPHA,
    TROPHY_COLOR, TROPHY_GLOW_ALPHA,
    FOG_UNEXPLORED_ALPHA, FOG_EXPLORED_BASE_ALPHA,
    GRID_ALPHA,
)
from sim.world import WorldState
from sim.physiology import OrganismState

# Lazy import pygame so headless works
_pygame = None


def _pg():
    global _pygame
    if _pygame is None:
        import pygame
        _pygame = pygame
    return _pygame


class PygameRenderer:
    """Renders the world, overlays, agent, hunters, trophy, fog, and HUD."""

    OVERLAY_NONE = 0
    OVERLAY_TEMPERATURE = 1
    OVERLAY_WATER = 2
    OVERLAY_VEGETATION = 3
    OVERLAY_WILDLIFE = 4
    OVERLAY_SHELTER = 5
    OVERLAY_ELEVATION = 6

    def __init__(self, config: Config, world: WorldState):
        pg = _pg()
        pg.init()

        self.config = config
        self.world = world
        self.cell_size = config.render.cell_size
        self.hud_width = config.render.hud_width
        self.fps = config.render.fps
        self.trail_length = config.render.trail_length

        self.map_w = world.w * self.cell_size
        self.map_h = world.h * self.cell_size
        self.screen_w = self.map_w + self.hud_width
        self.screen_h = self.map_h

        self.screen = pg.display.set_mode((self.screen_w, self.screen_h))
        pg.display.set_caption("Earth2D Survival")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("monospace", 14)
        self.font_small = pg.font.SysFont("monospace", 11)

        self.overlay_mode = self.OVERLAY_NONE
        self.trail: list[tuple[int, int]] = []

        # Fog of war tracking
        self.explored_cells: set[tuple[int, int]] = set()
        self.fog_enabled = config.fog_of_war.enabled
        self.fog_radius = config.fog_of_war.visibility_radius
        self.fog_dim_factor = config.fog_of_war.explored_dim_factor

        # Animation state
        self._frame_count = 0

        # Pre-render terrain base surface
        self._terrain_surface = pg.Surface((self.map_w, self.map_h))
        self._render_terrain_base()

        # Pre-render grid surface
        self._grid_surface = pg.Surface((self.map_w, self.map_h), pg.SRCALPHA)
        self._render_grid()

    def _render_terrain_base(self) -> None:
        pg = _pg()
        for y in range(self.world.h):
            for x in range(self.world.w):
                idx = self.world.terrain_id[y, x]
                color = TERRAIN_COLORS_BY_IDX[idx]
                rect = pg.Rect(
                    x * self.cell_size, y * self.cell_size,
                    self.cell_size, self.cell_size,
                )
                pg.draw.rect(self._terrain_surface, color, rect)

    def _render_grid(self) -> None:
        """Pre-render subtle grid lines."""
        pg = _pg()
        cs = self.cell_size
        grid_color = (0, 0, 0, GRID_ALPHA)
        for x in range(0, self.map_w, cs):
            pg.draw.line(self._grid_surface, grid_color, (x, 0), (x, self.map_h))
        for y in range(0, self.map_h, cs):
            pg.draw.line(self._grid_surface, grid_color, (0, y), (self.map_w, y))

    def handle_events(self) -> bool:
        """Process pygame events. Returns False if quit requested."""
        pg = _pg()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return False
                if event.key == pg.K_0:
                    self.overlay_mode = self.OVERLAY_NONE
                elif event.key == pg.K_1:
                    self.overlay_mode = self.OVERLAY_TEMPERATURE
                elif event.key == pg.K_2:
                    self.overlay_mode = self.OVERLAY_WATER
                elif event.key == pg.K_3:
                    self.overlay_mode = self.OVERLAY_VEGETATION
                elif event.key == pg.K_4:
                    self.overlay_mode = self.OVERLAY_WILDLIFE
                elif event.key == pg.K_5:
                    self.overlay_mode = self.OVERLAY_SHELTER
                elif event.key == pg.K_6:
                    self.overlay_mode = self.OVERLAY_ELEVATION
        return True

    def render(
        self,
        organism: OrganismState,
        step: int,
        time_info: dict,
        last_action: str = "",
        last_events: list[str] | None = None,
        hunters: list[dict] | None = None,
        trophy_pos: tuple[int, int] | None = None,
    ) -> None:
        """Render one frame."""
        pg = _pg()
        self._frame_count += 1
        self.screen.fill((0, 0, 0))

        # 1. Terrain base
        self.screen.blit(self._terrain_surface, (0, 0))

        # 2. Grid lines
        self.screen.blit(self._grid_surface, (0, 0))

        # 3. Overlay (if active)
        if self.overlay_mode != self.OVERLAY_NONE:
            self._draw_overlay()

        # 4. Trophy (drawn before fog so it's hidden in fog)
        if trophy_pos:
            self._draw_trophy(trophy_pos[0], trophy_pos[1], organism.x, organism.y)

        # 5. Hunters (drawn before fog so they're hidden in fog)
        if hunters:
            self._draw_hunters(hunters, organism.x, organism.y)

        # 6. Fog of war (on top of everything except agent/trail and HUD)
        if self.fog_enabled:
            self._draw_fog_of_war(organism.x, organism.y)

        # 7. Trail
        self.trail.append((organism.x, organism.y))
        if len(self.trail) > self.trail_length:
            self.trail = self.trail[-self.trail_length:]
        self._draw_trail()

        # 8. Agent (pulsing, on top of fog)
        self._draw_agent_pulsing(organism.x, organism.y)

        # 9. HUD
        self._draw_hud(organism, step, time_info, last_action, last_events or [],
                       hunters=hunters, trophy_pos=trophy_pos)

        pg.display.flip()
        self.clock.tick(self.fps)

    def _draw_overlay(self) -> None:
        pg = _pg()
        overlay_surf = pg.Surface((self.map_w, self.map_h), pg.SRCALPHA)
        cs = self.cell_size
        w = self.world

        for y in range(w.h):
            for x in range(w.w):
                if self.overlay_mode == self.OVERLAY_TEMPERATURE:
                    color = temperature_color(w.air_temperature_c[y, x])
                elif self.overlay_mode == self.OVERLAY_WATER:
                    color = water_color(w.water_availability[y, x])
                elif self.overlay_mode == self.OVERLAY_VEGETATION:
                    color = vegetation_color(w.vegetation_biomass[y, x])
                elif self.overlay_mode == self.OVERLAY_WILDLIFE:
                    color = wildlife_color(w.wildlife_risk[y, x])
                elif self.overlay_mode == self.OVERLAY_SHELTER:
                    color = shelter_color(w.shelter_quality[y, x])
                elif self.overlay_mode == self.OVERLAY_ELEVATION:
                    color = elevation_color(w.elevation[y, x])
                else:
                    continue

                rect = pg.Rect(x * cs, y * cs, cs, cs)
                pg.draw.rect(overlay_surf, (*color, 160), rect)

        self.screen.blit(overlay_surf, (0, 0))

    def _draw_fog_of_war(self, org_x: int, org_y: int) -> None:
        """Draw fog of war overlay. Updates explored set."""
        pg = _pg()
        R = self.fog_radius
        R_sq = R * R

        # Update explored cells (Euclidean circle)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dx * dx + dy * dy <= R_sq:
                    cx, cy = org_x + dx, org_y + dy
                    if 0 <= cx < self.world.w and 0 <= cy < self.world.h:
                        self.explored_cells.add((cx, cy))

        fog_surf = pg.Surface((self.map_w, self.map_h), pg.SRCALPHA)
        cs = self.cell_size

        for y in range(self.world.h):
            for x in range(self.world.w):
                dist_sq = (x - org_x) ** 2 + (y - org_y) ** 2
                if dist_sq <= R_sq:
                    continue  # fully visible
                elif (x, y) in self.explored_cells:
                    # Previously seen: dimmed
                    alpha = int(255 * (1.0 - self.fog_dim_factor))
                    pg.draw.rect(fog_surf, (0, 0, 0, alpha),
                                 (x * cs, y * cs, cs, cs))
                else:
                    # Never seen: fully dark
                    pg.draw.rect(fog_surf, (15, 15, 20, FOG_UNEXPLORED_ALPHA),
                                 (x * cs, y * cs, cs, cs))

        self.screen.blit(fog_surf, (0, 0))

    def _draw_hunters(self, hunters: list[dict], org_x: int, org_y: int) -> None:
        """Draw hunter NPCs with detection radius rings."""
        pg = _pg()
        cs = self.cell_size

        for h in hunters:
            hx = h["pos"]["x"]
            hy = h["pos"]["y"]

            # If fog enabled, only draw if in visible range
            if self.fog_enabled:
                dist_sq = (hx - org_x) ** 2 + (hy - org_y) ** 2
                if dist_sq > self.fog_radius ** 2:
                    continue

            cx = hx * cs + cs // 2
            cy = hy * cs + cs // 2

            # Detection radius ring (semi-transparent red)
            det_radius = h.get("detection_radius", 5)
            radius_px = det_radius * cs
            ring_surf = pg.Surface((radius_px * 2 + 4, radius_px * 2 + 4), pg.SRCALPHA)
            pg.draw.circle(
                ring_surf, (255, 50, 50, HUNTER_RADIUS_ALPHA),
                (radius_px + 2, radius_px + 2), radius_px, 2,
            )
            self.screen.blit(ring_surf, (cx - radius_px - 2, cy - radius_px - 2))

            # Hunter body
            is_chasing = h.get("is_chasing", False)
            color = HUNTER_CHASING_COLOR if is_chasing else HUNTER_COLOR
            pg.draw.circle(self.screen, color, (cx, cy), cs // 2 + 1)
            pg.draw.circle(self.screen, (0, 0, 0), (cx, cy), cs // 2 + 1, 1)

            # X marker on hunter
            half = cs // 3
            pg.draw.line(self.screen, (255, 255, 255),
                         (cx - half, cy - half), (cx + half, cy + half), 2)
            pg.draw.line(self.screen, (255, 255, 255),
                         (cx - half, cy + half), (cx + half, cy - half), 2)

    def _draw_trophy(self, tx: int, ty: int, org_x: int, org_y: int) -> None:
        """Draw the trophy with a pulsing golden glow."""
        pg = _pg()
        cs = self.cell_size

        # If fog enabled, only draw if visible
        if self.fog_enabled:
            dist_sq = (tx - org_x) ** 2 + (ty - org_y) ** 2
            if dist_sq > self.fog_radius ** 2:
                return

        cx = tx * cs + cs // 2
        cy = ty * cs + cs // 2

        # Pulsing glow
        pulse = 0.5 + 0.5 * math.sin(self._frame_count * 0.1)
        glow_radius = int(cs * (1.0 + pulse * 0.5))
        glow_surf = pg.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pg.SRCALPHA)
        glow_alpha = int(30 + 40 * pulse)
        pg.draw.circle(glow_surf, (255, 255, 150, glow_alpha),
                        (glow_radius + 2, glow_radius + 2), glow_radius)
        self.screen.blit(glow_surf, (cx - glow_radius - 2, cy - glow_radius - 2))

        # Trophy marker: golden circle
        pg.draw.circle(self.screen, TROPHY_COLOR, (cx, cy), cs // 2 + 1)
        pg.draw.circle(self.screen, (0, 0, 0), (cx, cy), cs // 2 + 1, 1)

        # Star inside
        half = cs // 4
        pg.draw.line(self.screen, (255, 255, 255), (cx, cy - half), (cx, cy + half), 2)
        pg.draw.line(self.screen, (255, 255, 255), (cx - half, cy), (cx + half, cy), 2)

    def _draw_agent_pulsing(self, org_x: int, org_y: int) -> None:
        """Draw the agent with a pulsing glow effect."""
        pg = _pg()
        cs = self.cell_size
        cx = org_x * cs + cs // 2
        cy = org_y * cs + cs // 2

        pulse = 0.5 + 0.5 * math.sin(self._frame_count * 0.15)
        radius = cs // 2 + 1 + int(pulse * 2)

        # Outer glow
        glow_r = radius * 2
        glow_surf = pg.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pg.SRCALPHA)
        pg.draw.circle(glow_surf, (255, 255, 100, 35),
                        (glow_r + 2, glow_r + 2), glow_r)
        self.screen.blit(glow_surf, (cx - glow_r - 2, cy - glow_r - 2))

        # Agent circle
        r_val = int(255 - 55 * pulse)
        g_val = int(255 - 55 * pulse)
        b_val = int(50 * pulse)
        pg.draw.circle(self.screen, (r_val, g_val, b_val), (cx, cy), radius)
        pg.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius, 1)

    def _draw_trail(self) -> None:
        pg = _pg()
        cs = self.cell_size
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            color = (255, 200, 0)
            cx = tx * cs + cs // 2
            cy = ty * cs + cs // 2
            radius = max(1, cs // 4)
            pg.draw.circle(self.screen, color, (cx, cy), radius)

    def _draw_hud(
        self,
        org: OrganismState,
        step: int,
        time_info: dict,
        last_action: str,
        events: list[str],
        hunters: list[dict] | None = None,
        trophy_pos: tuple[int, int] | None = None,
    ) -> None:
        pg = _pg()
        hud_x = self.map_w
        hud_rect = pg.Rect(hud_x, 0, self.hud_width, self.screen_h)
        pg.draw.rect(self.screen, HUD_BG, hud_rect)

        y_off = 10

        def text(txt: str, color: tuple = HUD_TEXT, small: bool = False) -> None:
            nonlocal y_off
            f = self.font_small if small else self.font
            surf = f.render(txt, True, color)
            self.screen.blit(surf, (hud_x + 10, y_off))
            y_off += surf.get_height() + 2

        def bar(label: str, value: float, max_val: float, color: tuple) -> None:
            nonlocal y_off
            text(f"{label}: {value:.1f}/{max_val:.0f}", small=True)
            bar_w = self.hud_width - 20
            bar_h = 12
            pg.draw.rect(self.screen, HUD_BAR_BG, (hud_x + 10, y_off, bar_w, bar_h))
            fill_w = int(bar_w * max(0, min(1, value / max_val)))
            pg.draw.rect(self.screen, color, (hud_x + 10, y_off, fill_w, bar_h))
            y_off += bar_h + 4

        # Title
        text("EARTH2D SURVIVAL", (255, 220, 100))
        y_off += 3

        # Time
        day = time_info.get("day_of_year", 0)
        hour = time_info.get("hour", 0)
        text(f"Step: {step}  Day: {day}  Hr: {hour:02d}")
        text(f"Pos: ({org.x}, {org.y})")
        y_off += 3

        # Status
        if org.alive:
            status_color = (100, 255, 100)
            status_text = "ALIVE"
        else:
            status_color = (255, 60, 60)
            status_text = f"DEAD: {org.cause_of_death.value}"
        text(status_text, status_color)
        y_off += 3

        # Bars
        bar("Hydration", org.hydration, 100, BAR_HYDRATION)
        bar("Energy", org.energy, 100, BAR_ENERGY)
        bar("Fatigue", org.fatigue, 100, BAR_FATIGUE)
        bar("Injury", org.injury, 100, BAR_INJURY)
        bar("Infection", org.infection, 100, BAR_INFECTION)

        # Core temp
        temp_color = HUD_TEXT
        if org.core_temp_c < 35:
            temp_color = (100, 150, 255)
        elif org.core_temp_c > 39:
            temp_color = (255, 100, 50)
        text(f"Core Temp: {org.core_temp_c:.1f}C", temp_color)
        text(f"Shelter: {'YES' if org.has_shelter else 'no'}", small=True)
        y_off += 5

        # Hunter info
        if hunters:
            text(f"Hunters: {len(hunters)} on map", (255, 120, 120))
            if org.alive:
                nearest = min(
                    hunters,
                    key=lambda h: abs(h["pos"]["x"] - org.x) + abs(h["pos"]["y"] - org.y),
                )
                nd = abs(nearest["pos"]["x"] - org.x) + abs(nearest["pos"]["y"] - org.y)
                chase_txt = " CHASING!" if nearest.get("is_chasing") else ""
                text(f"  Nearest: {nd} cells{chase_txt}", (255, 150, 150), small=True)
            y_off += 3

        # Trophy info
        if trophy_pos:
            tx, ty = trophy_pos
            td = abs(tx - org.x) + abs(ty - org.y)
            if td < 5:
                text("Trophy: VERY CLOSE!", (255, 255, 100))
            elif td < 15:
                text("Trophy: Close", (255, 220, 100))
            else:
                text("Trophy: Searching...", (180, 180, 100))
            y_off += 3

        # Last action
        text(f"Action: {last_action}", (180, 220, 255))
        y_off += 3

        # Events
        text("Events:", (180, 180, 180))
        for ev in events[-5:]:
            text(f"  {ev[:38]}", (160, 160, 160), small=True)

        # Overlay legend at bottom
        y_off = self.screen_h - 100
        text("Overlays (keys):", (140, 140, 150))
        text("0:None 1:Temp 2:Water", (120, 120, 130), small=True)
        text("3:Veg 4:Wildlife 5:Shelter", (120, 120, 130), small=True)
        text("6:Elevation  ESC:Quit", (120, 120, 130), small=True)

        overlay_names = ["None", "Temperature", "Water", "Vegetation",
                         "Wildlife", "Shelter", "Elevation"]
        text(f"Current: {overlay_names[self.overlay_mode]}", (200, 200, 100), small=True)

    def close(self) -> None:
        _pg().quit()
