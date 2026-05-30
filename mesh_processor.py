"""
mesh_processor.py — Kinetic Matrix Projection Layer
====================================================
Handles all post-ingestion geometry work:
  • Vault mesh normalization into clean presentation viewports
  • Structural breakdown (explosion) vector computation
  • Wireframe edge index buffer extraction
  • Telemetry path positioning — real-time 4×4 matrix generation
    from TelemetryEngine interpolated position + orientation
"""
import logging
import io
from typing import List, Dict, Any, Optional

import numpy as np
import trimesh
import trimesh.repair

from orchestrator import AssetIntent, ComponentNode, AssetClass
from mesh_builder import (
    IngestionPipeline, VaultLoader, TelemetryEngine,
    _repair, _extract_wireframe_edges, _compute_explosion_vector,
)

logger = logging.getLogger("MeshProcessor")

# Desired maximum dimension of a normalised asset in scene units
NORMALISE_TARGET = 0.8


class GeometryProcessor:
    """
    Single entry-point for all geometry processing.
    Unified handler for mechanical trees, terrain, and telemetry payloads.
    """

    def __init__(self):
        self.pipeline   = IngestionPipeline()      # legacy + terrain
        self.vault      = VaultLoader()
        self.telemetry  = TelemetryEngine()

    # =========================================================================
    # PUBLIC — main orchestration endpoint (called by main.py)
    # =========================================================================

    async def build_3d_simulation_payload(self, intent: AssetIntent) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "asset_name": intent.asset_name,
            "asset_type": intent.asset_type.value,
            "components": [],
            "telemetry":  None,   # filled for TELEMETRY mode
        }

        if intent.asset_type == AssetClass.TERRAIN:
            payload["components"] = self._build_terrain(intent)

        elif intent.asset_type == AssetClass.TELEMETRY:
            payload["components"], payload["telemetry"] = self._build_telemetry(intent)

        else:   # MECHANICAL
            payload["components"] = self._build_mechanical(intent)

        return payload

    # =========================================================================
    # TERRAIN
    # =========================================================================

    def _build_terrain(self, intent: AssetIntent) -> List[Dict[str, Any]]:
        config = intent.terrain_config
        if config is None:
            from orchestrator import TerrainConfig
            config = TerrainConfig()

        mesh = self.pipeline.generate_procedural_terrain(config)
        mesh = self.normalize_and_repair(mesh)

        return [{
            "name":             "terrain_surface",
            "primitive_type":   "terrain_heightmap",
            "vertices":         mesh.vertices.tolist(),
            "faces":            mesh.faces.tolist(),
            "normals":          mesh.vertex_normals.tolist(),
            "color":            [46, 125, 50, 255],
            "global_transform": np.eye(4).flatten().tolist(),
            "explosion_vector": [0.0, 0.0, 0.0],
            "wireframe_edges":  _extract_wireframe_edges(mesh),
        }]

    # =========================================================================
    # MECHANICAL (legacy component tree)
    # =========================================================================

    def _build_mechanical(self, intent: AssetIntent) -> List[Dict[str, Any]]:
        if intent.structural_tree:
            return self.process_mechanical_tree(intent.structural_tree)
        # Empty tree — single unit cube as absolute last resort
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        _repair(mesh)
        return [{
            "name":             "default_primitive",
            "primitive_type":   "box",
            "vertices":         mesh.vertices.tolist(),
            "faces":            mesh.faces.tolist(),
            "normals":          mesh.vertex_normals.tolist(),
            "color":            [128, 128, 128, 255],
            "global_transform": np.eye(4).flatten().tolist(),
            "explosion_vector": [0.0, 1.0, 0.0],
            "wireframe_edges":  _extract_wireframe_edges(mesh),
        }]

    # =========================================================================
    # TELEMETRY — vault load + trajectory computation
    # =========================================================================

    def _build_telemetry(self, intent: AssetIntent):
        """
        Returns (components, telemetry_block).

        components    — normalised vault/procedural mesh parts (same format as
                        mechanical mode; the streaming loop animates them along
                        the path via per-frame orientation matrices).

        telemetry_block — dict containing:
            scene_trajectory : List[[x,y,z]]  pre-projected arc in scene space
            total_distance_km: float
            estimated_duration_h: float
            origin_label     : str
            destination_label: str
            speed_kmh        : float
        """
        # ── 1. Load visual mesh ──────────────────────────────────────────
        vt = intent.visual_target
        if vt:
            raw_parts, source = self.vault.load(
                vt.vault_key, vt.fallback_assembly, vt.scale_hint
            )
            logger.info(f"Visual asset source: {source}")
        else:
            # Fallback: generic aircraft procedural assembly
            from scraper import VaultLoader as VL
            raw_parts, _ = VL().load("generic_aircraft", "aircraft", [40.0, 12.0, 35.0])

        # ── 2. Normalise the assembled part set into a compact viewport unit ─
        components = self._normalise_parts(raw_parts)

        # ── 3. Pre-compute scene-space trajectory ────────────────────────
        route = intent.telemetry_route
        scene_traj: List[List[float]] = []
        telemetry_block: Optional[Dict[str, Any]] = None

        if route:
            # Ensure trajectory is populated (may already be from orchestrator)
            if not route.trajectory:
                from orchestrator import _sample_great_circle
                route.trajectory = _sample_great_circle(
                    route.origin.lat, route.origin.lon,
                    route.destination.lat, route.destination.lon,
                    route.cruise_altitude_m, n_points=120,
                )

            scene_traj = self.telemetry.build_scene_trajectory(route)

            telemetry_block = {
                "scene_trajectory":    scene_traj,
                "total_distance_km":   route.total_distance_km,
                "estimated_duration_h": route.estimated_duration_h,
                "origin_label":        route.origin.label or "Origin",
                "destination_label":   route.destination.label or "Destination",
                "speed_kmh":           route.speed_kmh,
                "path_type":           route.path_type.value,
                "waypoint_count":      len(scene_traj),
            }

        return components, telemetry_block

    # =========================================================================
    # Normalisation — scales + centres a multi-part assembly
    # =========================================================================

    def _normalise_parts(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scales the entire multi-part assembly so its combined bounding box
        fits within NORMALISE_TARGET scene units, centred at the origin.
        Preserves relative positions between parts.
        """
        if not parts:
            return parts

        # Collect all vertices to find global bounding box
        all_verts = np.vstack([np.array(p["vertices"]) for p in parts
                                if p.get("vertices")])
        if len(all_verts) == 0:
            return parts

        bbox_min = all_verts.min(axis=0)
        bbox_max = all_verts.max(axis=0)
        center   = (bbox_min + bbox_max) / 2.0
        extents  = bbox_max - bbox_min
        max_dim  = extents.max()
        scale    = NORMALISE_TARGET / max_dim if max_dim > 1e-6 else 1.0

        normalised: List[Dict[str, Any]] = []
        for part in parts:
            verts = np.array(part["vertices"])
            # Centre then scale
            verts = (verts - center) * scale
            norms = np.array(part.get("normals", []))

            # Update the stored 4×4 global transform to include the same shift
            T      = np.array(part.get("global_transform",
                                       np.eye(4).flatten().tolist())).reshape(4, 4)
            T[0:3, 3] = (T[0:3, 3] - center) * scale

            # Recompute explosion vector from new centroid
            centroid = verts.mean(axis=0)
            n        = np.linalg.norm(centroid)
            exp_vec  = (centroid / n).tolist() if n > 1e-5 else [0.0, 1.0, 0.0]

            normalised.append({
                **part,
                "vertices":         verts.tolist(),
                "normals":          norms.tolist() if len(norms) > 0 else part.get("normals", []),
                "global_transform": T.flatten().tolist(),
                "explosion_vector": exp_vec,
            })

        return normalised

    # =========================================================================
    # Legacy API — mechanical component tree traversal
    # =========================================================================

    def process_mechanical_tree(
        self,
        node: ComponentNode,
        parent_transform: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if parent_transform is None:
            parent_transform = np.eye(4)

        local_T  = self.build_component_transform(node.relative_position,
                                                   node.relative_rotation)
        global_T = parent_transform @ local_T

        mesh = self.pipeline.generate_procedural_primitive(node.primitive_type, node.scale)
        mesh = self.normalize_and_repair(mesh)

        vertices = mesh.vertices.tolist()
        faces    = mesh.faces.tolist()
        normals  = mesh.vertex_normals.tolist() if len(mesh.vertex_normals) > 0 else []
        wireframe = _extract_wireframe_edges(mesh)

        global_pos = global_T[0:3, 3]
        norm       = np.linalg.norm(global_pos)
        if norm > 1e-4:
            exp_vec = (global_pos / norm).tolist()
        else:
            loc = np.array(node.relative_position)
            ln  = np.linalg.norm(loc)
            exp_vec = (loc / ln).tolist() if ln > 1e-4 else [0.0, 1.0, 0.0]

        _COLORS = {
            "shaft":    [180, 180, 190, 255],
            "cylinder": [150, 170, 185, 255],
            "gear":     [218, 165,  32, 255],
            "box":      [120, 150, 130, 255],
            "cube":     [120, 150, 130, 255],
            "sphere":   [200, 100, 100, 255],
            "wing":     [180, 190, 200, 255],
            "rotor":    [ 40,  40,  40, 255],
        }
        color = _COLORS.get(node.primitive_type.lower(), [160, 160, 160, 255])

        result = [{
            "name":             node.name,
            "primitive_type":   node.primitive_type,
            "vertices":         vertices,
            "faces":            faces,
            "normals":          normals,
            "color":            color,
            "global_transform": global_T.flatten().tolist(),
            "explosion_vector": exp_vec,
            "wireframe_edges":  wireframe,
        }]

        for child in node.children:
            result.extend(self.process_mechanical_tree(child, global_T))

        return result

    # =========================================================================
    # Real-time orientation matrix for streaming loop (telemetry mode)
    # =========================================================================

    def compute_path_transform(self, scene_trajectory: List[List[float]],
                                t: float,
                                base_rotation_angle: float = 0.0) -> np.ndarray:
        """
        Returns a 4×4 homogeneous transform matrix that:
          • translates the vehicle to its current arc position at time t
          • orients it to face the path tangent (look-ahead direction)
          • applies an optional global Y-rotation for the idle spin animation

        t  ∈ [0, 1] — normalised progress along the route.
        """
        pos, fwd = self.telemetry.interpolate_position(scene_trajectory, t)
        M        = self.telemetry.build_orientation_matrix(pos, fwd)

        # Optional gentle idle yaw animation (cosmetic global spin)
        if abs(base_rotation_angle) > 1e-6:
            cos_a = np.cos(base_rotation_angle)
            sin_a = np.sin(base_rotation_angle)
            Ry = np.array([
                [ cos_a, 0.0, sin_a, 0.0],
                [   0.0, 1.0,   0.0, 0.0],
                [-sin_a, 0.0, cos_a, 0.0],
                [   0.0, 0.0,   0.0, 1.0],
            ])
            M = M @ Ry

        return M

    # =========================================================================
    # Utility helpers (kept for legacy callers)
    # =========================================================================

    def normalize_and_repair(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if mesh.is_empty:
            return mesh
        _repair(mesh)
        mesh.apply_translation(-mesh.centroid)
        extents = mesh.extents
        max_dim = max(extents) if len(extents) > 0 else 0
        if max_dim > 1e-5:
            mesh.apply_scale(1.0 / max_dim)
        logger.info(f"Normalized mesh: V={len(mesh.vertices)} F={len(mesh.faces)}")
        return mesh

    def parse_mesh_from_bytes(self, data: bytes,
                               file_ext: str = "stl") -> Optional[trimesh.Trimesh]:
        try:
            mesh = trimesh.load(io.BytesIO(data), file_type=file_ext)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            return mesh
        except Exception as e:
            logger.error(f"parse_mesh_from_bytes failed: {e}")
            return None

    def calculate_euler_matrix(self, rot: List[float]) -> np.ndarray:
        roll, pitch, yaw = rot
        Rx = np.array([[1,0,0,0],
                       [0, np.cos(roll),-np.sin(roll),0],
                       [0, np.sin(roll), np.cos(roll),0],
                       [0,0,0,1]], dtype=float)
        Ry = np.array([[np.cos(pitch),0,np.sin(pitch),0],
                       [0,1,0,0],
                       [-np.sin(pitch),0,np.cos(pitch),0],
                       [0,0,0,1]], dtype=float)
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0,0],
                       [np.sin(yaw), np.cos(yaw),0,0],
                       [0,0,1,0],
                       [0,0,0,1]], dtype=float)
        return Rz @ Ry @ Rx

    def build_component_transform(self, pos: List[float],
                                   rot: List[float]) -> np.ndarray:
        T       = np.eye(4)
        T[0:3, 3] = pos
        return T @ self.calculate_euler_matrix(rot)
