"""
scraper.py — Hybrid Dual-Track Ingestion Pipeline
==================================================
Track 1 · Visuals   : Loads high-fidelity 3-D mesh from asset_vault/ by vault_key.
                      Falls back to a recognisable multi-part procedural assembly
                      (never a single primitive cube).
Track 2 · Telemetry : Pre-samples the geodesic arc trajectory and converts each
                      [lat, lon, alt] waypoint into a normalised 3-D scene vector
                      ready for WebSocket frame injection.
"""
import asyncio
import logging
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import httpx
import numpy as np
import trimesh
import trimesh.repair

from orchestrator import (
    TerrainConfig, ComponentNode, AssetClass,
    TelemetryRoute, VisualTarget, VehicleCategory,
)

logger = logging.getLogger("Scraper")

# Root of the pre-downloaded professional asset library
ASSET_VAULT = Path(__file__).parent / "asset_vault"
ASSET_VAULT.mkdir(exist_ok=True)

# Supported mesh extensions searched in priority order
_MESH_EXTENSIONS = [".glb", ".gltf", ".obj", ".stl", ".ply"]

# ---------------------------------------------------------------------------
# Colour palette — used by both vault meshes and procedural assemblies
# ---------------------------------------------------------------------------
PALETTE = {
    "fuselage":    [200, 210, 220, 255],   # cool aluminium
    "wing":        [180, 190, 200, 255],   # slightly darker
    "engine":      [80,  80,  80,  255],   # dark titanium
    "stabilizer":  [160, 170, 180, 255],
    "nacelle":     [60,  60,  60,  255],
    "hull":        [50,  80, 110, 255],    # deep navy
    "superstructure": [220, 220, 210, 255],
    "deck":        [140, 130, 100, 255],
    "crane":       [240, 180,  40, 255],   # hi-vis yellow
    "body":        [200, 200, 200, 255],
    "rotor":       [40,  40,  40,  255],
    "arm":         [120, 130, 140, 255],
    "shaft":       [180, 180, 190, 255],
    "gear":        [218, 165,  32, 255],
    "cylinder":    [150, 170, 185, 255],
    "box":         [120, 150, 130, 255],
    "sphere":      [200, 100, 100, 255],
    "default":     [160, 160, 160, 255],
}


# ===========================================================================
# TRACK 1 — VISUAL ASSET LOADING
# ===========================================================================

class VaultLoader:
    """
    Searches asset_vault/ for a mesh matching the given vault_key.
    Returns (mesh, source_label) where source_label is 'vault' or 'procedural'.
    """

    def load(self, vault_key: str, fallback_assembly: str,
             scale_hint: List[float]) -> Tuple[List[Dict[str, Any]], str]:
        mesh_parts = self._try_vault(vault_key)
        if mesh_parts:
            logger.info(f"Vault hit: loaded '{vault_key}' from asset_vault/")
            return mesh_parts, "vault"

        logger.info(f"Vault miss for '{vault_key}' — building procedural assembly: {fallback_assembly}")
        mesh_parts = self._build_assembly(fallback_assembly, scale_hint)
        return mesh_parts, "procedural"

    # ── Vault file search ──────────────────────────────────────────────────

    def _try_vault(self, vault_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Looks for vault_key.<ext> inside asset_vault/.
        Splits the loaded scene into named parts for the streaming pipeline.
        """
        for ext in _MESH_EXTENSIONS:
            candidate = ASSET_VAULT / f"{vault_key}{ext}"
            if candidate.exists():
                return self._load_file(candidate)
        return None

    def _load_file(self, path: Path) -> Optional[List[Dict[str, Any]]]:
        try:
            loaded = trimesh.load(str(path), force="mesh", process=False)
            parts: List[Dict[str, Any]] = []

            if isinstance(loaded, trimesh.Scene):
                for name, geom in loaded.geometry.items():
                    part = self._mesh_to_part(geom, name)
                    if part:
                        parts.append(part)
            elif isinstance(loaded, trimesh.Trimesh):
                part = self._mesh_to_part(loaded, path.stem)
                if part:
                    parts.append(part)

            return parts if parts else None
        except Exception as e:
            logger.error(f"Failed to load vault file {path}: {e}")
            return None

    def _mesh_to_part(self, mesh: trimesh.Trimesh,
                      name: str) -> Optional[Dict[str, Any]]:
        if mesh.is_empty or len(mesh.faces) == 0:
            return None
        _repair(mesh)
        color = _color_for_name(name)
        wireframe_edges = _extract_wireframe_edges(mesh)
        explosion_vec   = _compute_explosion_vector(mesh)
        return {
            "name":             name,
            "primitive_type":   "vault_mesh",
            "vertices":         mesh.vertices.tolist(),
            "faces":            mesh.faces.tolist(),
            "normals":          mesh.vertex_normals.tolist(),
            "color":            color,
            "global_transform": np.eye(4).flatten().tolist(),
            "explosion_vector": explosion_vec,
            "wireframe_edges":  wireframe_edges,
        }

    # ── Procedural high-fidelity assemblies ───────────────────────────────

    def _build_assembly(self, assembly_type: str,
                        scale_hint: List[float]) -> List[Dict[str, Any]]:
        builders = {
            "aircraft": self._aircraft,
            "fighter":  self._fighter_jet,
            "ship":     self._container_ship,
            "drone":    self._quadcopter,
            "rocket":   self._rocket,
            "train":    self._high_speed_train,
        }
        builder = builders.get(assembly_type, self._aircraft)
        return builder(scale_hint)

    # ── Aircraft ───────────────────────────────────────────────────────────

    def _aircraft(self, _scale: List[float]) -> List[Dict[str, Any]]:
        """Commercial wide-body aircraft: fuselage + delta wings + tail + engines."""
        parts: List[Dict[str, Any]] = []
        L = 1.0   # normalised fuselage half-length

        # Fuselage — tapered cylinder approximated by scaled ellipsoid-ish box
        fuselage = trimesh.creation.cylinder(radius=0.055, height=2*L, sections=36)
        parts.append(self._part(fuselage, "fuselage", [0, 0, 0], PALETTE["fuselage"]))

        # Wings — swept trapezoid via vertex extrusion
        for side, sx in [("left_wing", -1), ("right_wing", 1)]:
            wing = _make_swept_wing(span=0.72, root_chord=0.28, tip_chord=0.10,
                                     sweep_angle=32.0, thickness=0.012, side=sx)
            parts.append(self._part(wing, side, [sx*0.06, 0, 0.04], PALETTE["wing"]))

        # Horizontal stabilisers
        for side, sx in [("stab_left", -1), ("stab_right", 1)]:
            stab = _make_swept_wing(span=0.22, root_chord=0.12, tip_chord=0.05,
                                     sweep_angle=28.0, thickness=0.008, side=sx)
            parts.append(self._part(stab, side, [sx*0.025, 0, -L*0.86], PALETTE["stabilizer"]))

        # Vertical stabiliser
        vstab = _make_swept_wing(span=0.18, root_chord=0.14, tip_chord=0.06,
                                  sweep_angle=38.0, thickness=0.01, side=1)
        vstab.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
        parts.append(self._part(vstab, "vstab", [0, 0.04, -L*0.86], PALETTE["stabilizer"]))

        # Engines under wings (nacelle + intake ring)
        for name, pos in [("engine_L", [-0.28, -0.045, 0.08]),
                           ("engine_R", [ 0.28, -0.045, 0.08])]:
            nacelle = trimesh.creation.cylinder(radius=0.035, height=0.22, sections=24)
            intake  = trimesh.creation.annulus(r_min=0.030, r_max=0.038, height=0.012, sections=24)
            intake.apply_translation([0, 0, 0.115])
            eng = trimesh.util.concatenate([nacelle, intake])
            eng.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
            parts.append(self._part(eng, name, pos, PALETTE["engine"]))

        return parts

    # ── Fighter jet ────────────────────────────────────────────────────────

    def _fighter_jet(self, _scale: List[float]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        # Slender fuselage with nose cone
        fuselage = trimesh.creation.cylinder(radius=0.04, height=1.6, sections=24)
        nose_verts = np.array([[0,0,0.8],[0.04,0,0.55],[-0.04,0,0.55],
                                [0,0.04,0.55],[0,-0.04,0.55]])
        # simple cone cap
        nose = trimesh.creation.cone(radius=0.04, height=0.22, sections=20)
        nose.apply_translation([0, 0, 0.69])
        body = trimesh.util.concatenate([fuselage, nose])
        parts.append(self._part(body, "fuselage", [0,0,0], PALETTE["fuselage"]))

        # Delta wings (large, highly swept)
        for side, sx in [("wing_L", -1), ("wing_R", 1)]:
            wing = _make_swept_wing(span=0.55, root_chord=0.60, tip_chord=0.04,
                                     sweep_angle=55.0, thickness=0.010, side=sx)
            parts.append(self._part(wing, side, [sx*0.04, 0, 0.1], PALETTE["wing"]))

        # Twin vertical tails
        for side, sx in [("vtail_L", -1), ("vtail_R", 1)]:
            vtail = _make_swept_wing(span=0.16, root_chord=0.22, tip_chord=0.06,
                                      sweep_angle=42.0, thickness=0.008, side=1)
            vtail.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
            parts.append(self._part(vtail, side, [sx*0.04, 0.04, -0.55], PALETTE["stabilizer"]))

        # Twin afterburner nozzles
        for side, sx in [("nozzle_L", -1), ("nozzle_R", 1)]:
            nozzle = trimesh.creation.cylinder(radius=0.028, height=0.20, sections=20)
            nozzle.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
            parts.append(self._part(nozzle, side, [sx*0.04, 0, -0.85], PALETTE["nacelle"]))

        return parts

    # ── Container ship ─────────────────────────────────────────────────────

    def _container_ship(self, _scale: List[float]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        # Hull — wide flat box with chamfered bow approximation
        hull = trimesh.creation.box(extents=[0.22, 0.80, 0.12])
        bow_cut = trimesh.creation.cone(radius=0.115, height=0.14, sections=20)
        bow_cut.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
        bow_cut.apply_translation([0, 0.46, 0.01])
        try:
            hull_shaped = hull.difference(bow_cut)
            if not hull_shaped.is_empty:
                hull = hull_shaped
        except Exception:
            pass
        parts.append(self._part(hull, "hull", [0, 0, 0], PALETTE["hull"]))

        # Container stacks (3 rows × 4 columns)
        for row in range(3):
            for col in range(4):
                box = trimesh.creation.box(extents=[0.065, 0.13, 0.045])
                x = (col - 1.5) * 0.072
                y = (row - 1.0) * 0.15
                z = 0.08 + row * 0.048
                color = [
                    [200, 60, 60, 255], [60, 140, 200, 255],
                    [200, 160, 40, 255], [60, 180, 80, 255]
                ][col % 4]
                parts.append(self._part(box, f"container_{row}_{col}", [x, y, z], color))

        # Bridge / superstructure
        bridge = trimesh.creation.box(extents=[0.10, 0.08, 0.10])
        parts.append(self._part(bridge, "bridge", [0, -0.30, 0.11], PALETTE["superstructure"]))

        # Funnel
        funnel = trimesh.creation.cylinder(radius=0.018, height=0.06, sections=16)
        parts.append(self._part(funnel, "funnel", [0, -0.28, 0.19], [60, 60, 60, 255]))

        # Anchor cranes
        for sy in [0.22, -0.18]:
            crane = trimesh.creation.box(extents=[0.012, 0.05, 0.08])
            parts.append(self._part(crane, f"crane_{sy}", [0.10, sy, 0.10], PALETTE["crane"]))

        return parts

    # ── Quadcopter drone ───────────────────────────────────────────────────

    def _quadcopter(self, _scale: List[float]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        # Central hub — flat octagonal body
        hub = trimesh.creation.cylinder(radius=0.18, height=0.04, sections=8)
        parts.append(self._part(hub, "hub", [0, 0, 0], [60, 60, 70, 255]))

        # Camera gimbal underneath
        gimbal = trimesh.creation.sphere(radius=0.045)
        parts.append(self._part(gimbal, "gimbal", [0, 0, -0.04], [30, 30, 30, 255]))

        arm_positions = [
            ("arm_FL", -0.22,  0.22),
            ("arm_FR",  0.22,  0.22),
            ("arm_BL", -0.22, -0.22),
            ("arm_BR",  0.22, -0.22),
        ]
        for name, ax, ay in arm_positions:
            # Carbon-fibre arm tube
            arm = trimesh.creation.cylinder(radius=0.016, height=0.40, sections=12)
            angle = math.atan2(ay, ax)
            arm.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [0,1,0]))
            arm.apply_transform(trimesh.transformations.rotation_matrix(angle, [0,0,1]))
            parts.append(self._part(arm, name, [ax/2, ay/2, 0], PALETTE["arm"]))

            # Motor bell housing
            motor = trimesh.creation.cylinder(radius=0.030, height=0.022, sections=16)
            parts.append(self._part(motor, f"motor_{name}", [ax, ay, 0.015], [40,40,40,255]))

            # 3-blade propeller
            for blade_i in range(3):
                blade_angle = blade_i * (2*math.pi/3)
                blade = trimesh.creation.box(extents=[0.001, 0.24, 0.018])
                blade.apply_transform(
                    trimesh.transformations.rotation_matrix(blade_angle, [0,0,1]))
                parts.append(self._part(blade, f"blade_{name}_{blade_i}",
                                        [ax, ay, 0.030], PALETTE["rotor"]))

        # Landing legs
        for lx, ly in [(-0.12, 0.12), (0.12, 0.12), (-0.12, -0.12), (0.12, -0.12)]:
            leg = trimesh.creation.cylinder(radius=0.006, height=0.10, sections=8)
            leg.apply_transform(trimesh.transformations.rotation_matrix(0.25, [1,0,0]))
            parts.append(self._part(leg, f"leg_{lx}_{ly}", [lx, ly, -0.05],
                                    [80, 80, 80, 255]))

        return parts

    # ── Rocket ─────────────────────────────────────────────────────────────

    def _rocket(self, _scale: List[float]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        # Stage 1 booster
        s1 = trimesh.creation.cylinder(radius=0.08, height=0.70, sections=32)
        parts.append(self._part(s1, "stage1", [0, 0, -0.15], [240, 240, 240, 255]))

        # Stage 2 upper
        s2 = trimesh.creation.cylinder(radius=0.07, height=0.35, sections=32)
        parts.append(self._part(s2, "stage2", [0, 0, 0.40], [220, 220, 230, 255]))

        # Payload fairing (cone)
        fairing = trimesh.creation.cone(radius=0.07, height=0.24, sections=32)
        parts.append(self._part(fairing, "fairing", [0, 0, 0.655], [200, 210, 220, 255]))

        # 9 Merlin engines (grid)
        engine_offsets = [(math.cos(a)*0.055, math.sin(a)*0.055)
                          for a in np.linspace(0, 2*math.pi, 9, endpoint=False)]
        for i, (ex, ey) in enumerate(engine_offsets):
            nozzle = trimesh.creation.cylinder(radius=0.018, height=0.08, sections=16)
            parts.append(self._part(nozzle, f"engine_{i}", [ex, ey, -0.525], [60,60,60,255]))

        # Grid fins (4×)
        for a in [0, math.pi/2, math.pi, 3*math.pi/2]:
            fin = trimesh.creation.box(extents=[0.06, 0.003, 0.07])
            fin.apply_transform(trimesh.transformations.rotation_matrix(a, [0,0,1]))
            fin.apply_translation([0.085*math.cos(a), 0.085*math.sin(a), -0.08])
            parts.append(self._part(fin, f"gridfin_{int(math.degrees(a))}",
                                    [0,0,0], [80,80,80,255]))

        # Landing legs (4×)
        for a in [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]:
            leg = trimesh.creation.cylinder(radius=0.005, height=0.18, sections=8)
            leg.apply_transform(trimesh.transformations.rotation_matrix(0.42, [1,0,0]))
            leg.apply_transform(trimesh.transformations.rotation_matrix(a, [0,0,1]))
            leg.apply_translation([0.075*math.cos(a), 0.075*math.sin(a), -0.48])
            parts.append(self._part(leg, f"leg_{int(math.degrees(a))}", [0,0,0],
                                    [100,100,100,255]))

        return parts

    # ── High-speed train ───────────────────────────────────────────────────

    def _high_speed_train(self, _scale: List[float]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        car_positions = [-0.75, -0.25, 0.25, 0.75]
        for i, cy in enumerate(car_positions):
            body = trimesh.creation.box(extents=[0.18, 0.42, 0.14])
            parts.append(self._part(body, f"car_{i}_body", [0, cy, 0.04],
                                    [220, 30, 30, 255] if i in [0, 3] else [240, 240, 250, 255]))

            # Windows strip
            windows = trimesh.creation.box(extents=[0.19, 0.30, 0.04])
            parts.append(self._part(windows, f"car_{i}_windows", [0, cy, 0.06],
                                    [160, 200, 220, 200]))

            # Bogies (wheel assemblies)
            for bx in [-0.06, 0.06]:
                bogie = trimesh.creation.box(extents=[0.06, 0.10, 0.03])
                parts.append(self._part(bogie, f"car_{i}_bogie_{bx}", [bx, cy, -0.025],
                                        [60, 60, 60, 255]))
                for wx in [-0.04, 0.04]:
                    wheel = trimesh.creation.cylinder(radius=0.025, height=0.015, sections=20)
                    wheel.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[0,1,0]))
                    parts.append(self._part(wheel, f"wheel_{i}_{bx}_{wx}",
                                            [wx, cy, -0.038], [40,40,40,255]))

        # Nose cone (aerodynamic prow)
        nose = trimesh.creation.cone(radius=0.09, height=0.30, sections=20)
        nose.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2, [1,0,0]))
        parts.append(self._part(nose, "nose", [0, 0.97, 0.04], [220, 30, 30, 255]))

        return parts

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _part(mesh: trimesh.Trimesh, name: str,
              position: List[float], color: List[int]) -> Dict[str, Any]:
        """Repair, translate, then pack a mesh into the streaming dict format."""
        _repair(mesh)
        mesh.apply_translation(position)
        explosion_vec = _compute_explosion_vector(mesh)
        wireframe     = _extract_wireframe_edges(mesh)
        T             = np.eye(4)
        T[0:3, 3]     = position
        return {
            "name":             name,
            "primitive_type":   name.split("_")[0],
            "vertices":         mesh.vertices.tolist(),
            "faces":            mesh.faces.tolist(),
            "normals":          mesh.vertex_normals.tolist(),
            "color":            color,
            "global_transform": T.flatten().tolist(),
            "explosion_vector": explosion_vec,
            "wireframe_edges":  wireframe,
        }


# ===========================================================================
# TRACK 2 — TELEMETRY / GEODESIC PATH
# ===========================================================================

class TelemetryEngine:
    """
    Converts a TelemetryRoute's pre-sampled [lat, lon, alt] trajectory into
    normalised 3-D scene coordinates ready for the WebSocket streaming loop.

    Coordinate mapping
    ------------------
    The globe is projected onto a unit sphere of radius SCENE_RADIUS so it
    fits cleanly in the WebGL viewport without floating-point precision issues.
    """
    SCENE_RADIUS = 1.8   # world sphere radius in scene units
    EARTH_RADIUS = 6371.0  # km

    def build_scene_trajectory(self, route: TelemetryRoute) -> List[List[float]]:
        """
        Returns a list of [x, y, z] scene-space waypoints (length = len(route.trajectory)).
        """
        if not route.trajectory:
            return []

        scene_pts: List[List[float]] = []
        max_alt   = max(pt[2] for pt in route.trajectory) or 1.0

        for lat, lon, alt in route.trajectory:
            x, y, z = self._latlon_to_scene(lat, lon, alt, max_alt)
            scene_pts.append([round(x, 6), round(y, 6), round(z, 6)])

        return scene_pts

    def interpolate_position(self, scene_trajectory: List[List[float]],
                              t: float) -> Tuple[List[float], List[float]]:
        """
        Returns (position_xyz, forward_direction_xyz) at normalised time t ∈ [0, 1].
        forward_direction is the tangent vector — used to orient the vehicle mesh.
        """
        if not scene_trajectory:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]

        n   = len(scene_trajectory)
        idx = t * (n - 1)
        lo  = min(int(idx), n - 2)
        hi  = lo + 1
        frac = idx - lo

        p0 = np.array(scene_trajectory[lo])
        p1 = np.array(scene_trajectory[hi])
        pos = (p0 + frac * (p1 - p0)).tolist()

        direction = p1 - p0
        norm      = np.linalg.norm(direction)
        fwd       = (direction / norm).tolist() if norm > 1e-9 else [0.0, 0.0, 1.0]

        return pos, fwd

    def build_orientation_matrix(self, position: List[float],
                                  forward: List[float]) -> np.ndarray:
        """
        Builds a 4×4 homogeneous transform that positions and orients
        the vehicle mesh so it faces 'forward' at 'position'.
        """
        fwd = np.array(forward, dtype=float)
        fwd /= (np.linalg.norm(fwd) + 1e-12)

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-9:
            world_up = np.array([0.0, 0.0, 1.0])
            right    = np.cross(fwd, world_up)
            r_norm   = np.linalg.norm(right) + 1e-12
        right /= r_norm
        up = np.cross(right, fwd)

        M = np.eye(4)
        M[0:3, 0] = right
        M[0:3, 1] = up
        M[0:3, 2] = fwd
        M[0:3, 3] = position
        return M

    # ── Internal ──────────────────────────────────────────────────────────

    def _latlon_to_scene(self, lat: float, lon: float,
                          alt: float, max_alt: float) -> Tuple[float, float, float]:
        """
        Maps WGS-84 [lat, lon, alt] → unit-sphere scene coords.
        Altitude lifts the point radially above the sphere surface.
        """
        phi = math.radians(lat)
        lam = math.radians(lon)
        # Radial distance = base sphere + altitude fraction
        alt_frac = (alt / max_alt) * 0.25 if max_alt > 0 else 0.0
        r = self.SCENE_RADIUS + alt_frac

        x = r * math.cos(phi) * math.cos(lam)
        y = r * math.sin(phi)
        z = r * math.cos(phi) * math.sin(lam)
        return x, y, z


# ===========================================================================
# LEGACY — IngestionPipeline (kept for backwards compatibility with
# mesh_processor.py's process_mechanical_tree + terrain paths)
# ===========================================================================

class IngestionPipeline:
    """
    Original ingestion pipeline — now delegates telemetry/visual work
    to VaultLoader and TelemetryEngine. Retained for backwards compatibility.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 EventNine/2.0",
            "Accept":     "application/vnd.github.v3+json",
        }
        self.vault   = VaultLoader()
        self.telemetry = TelemetryEngine()

    # ── GitHub CAD scraping (unchanged) ───────────────────────────────────

    async def scrape_github_cad(self, query: str) -> Optional[bytes]:
        clean = query.replace(" ", "+")
        url   = f"https://api.github.com/search/code?q={clean}+extension:stl"
        try:
            async with httpx.AsyncClient(timeout=4.0, headers=self.headers) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    if items:
                        raw_url = (items[0].get("html_url", "")
                                   .replace("github.com", "raw.githubusercontent.com")
                                   .replace("/blob/", "/"))
                        if raw_url:
                            fr = await client.get(raw_url)
                            if fr.status_code == 200 and len(fr.content) > 100:
                                return fr.content
                elif resp.status_code == 403:
                    logger.warning("GitHub API rate-limited.")
        except Exception as e:
            logger.error(f"GitHub scrape error: {e}")
        return None

    async def fetch_mechanical_asset(self, queries: List[str]) -> bytes:
        for q in queries[:2]:
            data = await self.scrape_github_cad(q)
            if data:
                return data
        logger.info("No remote CAD found — switching to procedural.")
        return b""

    # ── Primitive generator (legacy mechanical tree) ───────────────────────

    def generate_procedural_primitive(self, primitive_type: str,
                                       scale: List[float]) -> trimesh.Trimesh:
        p    = primitive_type.lower()
        sx, sy, sz = scale
        try:
            if p in ("cube", "box"):
                mesh = trimesh.creation.box(extents=[sx, sy, sz])
            elif p in ("cylinder", "shaft"):
                mesh = trimesh.creation.cylinder(radius=(sx+sy)/4.0, height=sz, sections=32)
            elif p == "sphere":
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=max(sx,sy,sz)/2.0)
            elif p == "gear":
                mesh = self._make_gear(sx, sy, sz)
            elif p == "wing":
                mesh = _make_swept_wing(span=sx, root_chord=sz, tip_chord=sz*0.35,
                                        sweep_angle=28.0, thickness=sy, side=1)
            elif p == "rotor":
                mesh = self._make_rotor(sx, sz)
            else:
                mesh = trimesh.creation.box(extents=[sx, sy, sz])
        except Exception as e:
            logger.error(f"Primitive '{primitive_type}' failed: {e}")
            mesh = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        return mesh

    def _make_gear(self, sx: float, sy: float, sz: float) -> trimesh.Trimesh:
        num_teeth    = 16
        height       = sz * 0.4
        outer_radius = max(sx, sy) / 2.0
        inner_radius = outer_radius * 0.75
        hub          = trimesh.creation.cylinder(radius=inner_radius, height=height, sections=32)
        teeth        = []
        for i in range(num_teeth):
            angle  = i * (2*math.pi / num_teeth)
            tooth_w = outer_radius - inner_radius
            tooth_t = outer_radius * 0.18
            tooth   = trimesh.creation.box(extents=[tooth_w, tooth_t, height])
            tx = (inner_radius + tooth_w/2.0) * math.cos(angle)
            ty = (inner_radius + tooth_w/2.0) * math.sin(angle)
            c, s = math.cos(angle), math.sin(angle)
            T = np.eye(4)
            T[0:3, 3]     = [tx, ty, 0.0]
            T[0:3, 0:3]   = [[c,-s,0],[s,c,0],[0,0,1]]
            tooth.apply_transform(T)
            teeth.append(tooth)
        mesh = trimesh.util.concatenate([hub] + teeth)
        hole = trimesh.creation.cylinder(radius=inner_radius*0.35, height=height+0.1, sections=16)
        try:
            diff = mesh.difference(hole)
            if not diff.is_empty:
                mesh = diff
        except Exception:
            pass
        return mesh

    def _make_rotor(self, diameter: float, thickness: float) -> trimesh.Trimesh:
        blades = []
        for i in range(3):
            angle = i * (2*math.pi/3)
            blade = trimesh.creation.box(extents=[diameter*0.48, thickness*0.5, diameter*0.06])
            R = trimesh.transformations.rotation_matrix(angle, [0,0,1])
            blade.apply_transform(R)
            blades.append(blade)
        hub = trimesh.creation.cylinder(radius=diameter*0.06, height=thickness*0.8, sections=16)
        return trimesh.util.concatenate([hub] + blades)

    # ── Terrain generator (unchanged interface) ────────────────────────────

    def generate_procedural_terrain(self, config: TerrainConfig) -> trimesh.Trimesh:
        try:
            grid_size = 60
            width = height = 10.0
            x = np.linspace(-width/2, width/2, grid_size)
            y = np.linspace(-height/2, height/2, grid_size)
            X, Y = np.meshgrid(x, y)
            Z    = np.zeros_like(X)

            f  = config.frequency
            Z += (np.sin(X*f) + np.cos(Y*f)) * (config.max_height * 0.5)

            f2 = f * 2.5; a2 = config.max_height * 0.25 * config.roughness
            Z += np.sin(X*f2) * np.cos(Y*f2) * a2

            f3 = f * 6.0; a3 = config.max_height * 0.1 * config.roughness**2
            Z += np.sin(X*f3 + Y*f3) * a3

            # Octave stacking
            for o in range(2, config.octaves):
                fo = f * (2.0 ** o); ao = (config.max_height * 0.05) / (2.0 ** o)
                Z += np.sin(X*fo + Y*fo*0.7) * np.cos(Y*fo - X*fo*0.3) * ao * config.roughness

            Z = np.clip(Z, config.min_height, config.max_height)
            sea = config.min_height + (config.max_height - config.min_height) * config.sea_level
            Z   = np.where(Z < sea, sea, Z)

            vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            faces    = []
            for i in range(grid_size - 1):
                for j in range(grid_size - 1):
                    v0 = i*grid_size + j
                    v1, v2, v3 = v0+1, (i+1)*grid_size+j, (i+1)*grid_size+j+1
                    faces.extend([[v0,v1,v2], [v1,v3,v2]])

            mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
            mesh.fix_normals()
            return mesh
        except Exception as e:
            logger.error(f"Terrain synthesis failed: {e}")
            return trimesh.creation.box(extents=[10.0, 10.0, 0.1])


# ===========================================================================
# Shared geometry utilities
# ===========================================================================

def _repair(mesh: trimesh.Trimesh) -> None:
    """In-place: weld vertices, fix normals, fill simple holes."""
    try:
        mesh.process(validate=True)
    except Exception:
        pass
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass


def _extract_wireframe_edges(mesh: trimesh.Trimesh) -> List[List[int]]:
    """
    Returns the unique edge index pairs that form the structural wireframe.
    Uses trimesh's face_adjacency edges to expose only topological contours.
    """
    try:
        edges = mesh.edges_unique
        return edges.tolist()
    except Exception:
        return []


def _compute_explosion_vector(mesh: trimesh.Trimesh) -> List[float]:
    """Outward explosion direction from mesh centroid."""
    try:
        centroid = mesh.centroid
        norm     = np.linalg.norm(centroid)
        if norm > 1e-5:
            return (centroid / norm).tolist()
        # Use bounding-box centre direction as fallback
        bb_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        n2 = np.linalg.norm(bb_center)
        if n2 > 1e-5:
            return (bb_center / n2).tolist()
    except Exception:
        pass
    return [0.0, 1.0, 0.0]


def _color_for_name(name: str) -> List[int]:
    """Semantic color lookup from part name."""
    n = name.lower()
    for key, color in PALETTE.items():
        if key in n:
            return color
    return PALETTE["default"]


def _make_swept_wing(span: float, root_chord: float, tip_chord: float,
                     sweep_angle: float, thickness: float, side: int) -> trimesh.Trimesh:
    """
    Builds a planar swept trapezoidal wing with finite thickness.
    side = +1 (right/port) or −1 (left/starboard).
    sweep_angle in degrees.
    """
    sweep_rad   = math.radians(sweep_angle)
    sweep_offset = span * math.tan(sweep_rad)   # tip leading-edge X offset

    sx = side
    # 8 vertices: 4 top face + 4 bottom face
    # Wing lies in XZ plane; Y is thickness axis
    half_t = thickness / 2.0
    verts = np.array([
        # bottom face (y = -half_t)
        [0.0,             -half_t,  0.0],             # root LE
        [root_chord,      -half_t,  0.0],             # root TE
        [sx*span + sweep_offset, -half_t, sx*span],   # tip LE
        [sx*span + sweep_offset + tip_chord, -half_t, sx*span],  # tip TE
        # top face (y = +half_t)
        [0.0,             half_t,  0.0],
        [root_chord,      half_t,  0.0],
        [sx*span + sweep_offset, half_t, sx*span],
        [sx*span + sweep_offset + tip_chord, half_t, sx*span],
    ], dtype=float)

    faces = np.array([
        # bottom
        [0,1,2], [1,3,2],
        # top
        [4,6,5], [5,6,7],
        # leading edge
        [0,4,2], [4,6,2],
        # trailing edge
        [1,3,5], [3,7,5],
        # root
        [0,1,4], [1,5,4],
        # tip
        [2,6,3], [3,6,7],
    ], dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _repair(mesh)
    return mesh
