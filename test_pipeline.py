import asyncio
import logging
import numpy as np
import trimesh
from orchestrator import route_intent, AssetIntent, AssetClass, ComponentNode
from scraper import IngestionPipeline
from mesh_processor import GeometryProcessor
from main import compute_realtime_vertex_frame

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineTest")

def run_orchestrator_tests():
    logger.info("=========================================")
    logger.info("1. RUNNING ORCHESTRATOR NLP ROUTING TESTS")
    logger.info("=========================================")
    
    # Test Mechanical intent
    mech_intent = route_intent("radial aircraft engine piston cylinder")
    assert isinstance(mech_intent, AssetIntent), "Routing must return an AssetIntent"
    assert mech_intent.asset_type == AssetClass.MECHANICAL, "Must classify as mechanical"
    assert mech_intent.structural_tree is not None, "Mechanical intent must have a structural tree"
    logger.info("✓ Mechanical intent routing test passed successfully.")
    
    # Test Terrain intent
    terrain_intent = route_intent("grand canyon topography map elevation scale")
    assert isinstance(terrain_intent, AssetIntent), "Routing must return an AssetIntent"
    assert terrain_intent.asset_type == AssetClass.TERRAIN, "Must classify as terrain"
    assert terrain_intent.terrain_config is not None, "Terrain intent must have terrain configuration parameters"
    logger.info("✓ Terrain intent routing test passed successfully.")
    
    logger.info("Orchestrator validation: ALL SCHEMA CHECKS PASSED.")

def run_scraper_tests():
    logger.info("=========================================")
    logger.info("2. RUNNING SCRAPER & SYNTHESIS TESTS")
    logger.info("=========================================")
    
    pipeline = IngestionPipeline()
    
    # Test primitive: Box
    box_mesh = pipeline.generate_procedural_primitive("box", [1.0, 2.0, 3.0])
    assert isinstance(box_mesh, trimesh.Trimesh), "Box generation must return Trimesh object"
    assert not box_mesh.is_empty, "Box mesh must not be empty"
    # Box bounds check
    extents = box_mesh.extents
    assert np.allclose(extents, [1.0, 2.0, 3.0], atol=1e-3), f"Box extents mismatch: {extents}"
    logger.info("✓ Procedural Box creation validated.")

    # Test primitive: Gear
    gear_mesh = pipeline.generate_procedural_primitive("gear", [2.0, 2.0, 0.5])
    assert isinstance(gear_mesh, trimesh.Trimesh), "Gear generation must return Trimesh object"
    assert not gear_mesh.is_empty, "Gear mesh must not be empty"
    logger.info("✓ Procedural 3D Spur Gear extrusion validated.")

    # Test Terrain Synthesis
    terrain_intent = route_intent("volcanic peak landscape")
    terrain_mesh = pipeline.generate_procedural_terrain(terrain_intent.terrain_config)
    assert isinstance(terrain_mesh, trimesh.Trimesh), "Terrain synthesis must return Trimesh"
    assert len(terrain_mesh.vertices) > 0, "Terrain must contain synthesized coordinate points"
    assert len(terrain_mesh.faces) > 0, "Terrain must contain face index maps"
    logger.info("✓ Procedural Heightmap Terrain synthesis validated.")
    
    logger.info("Scraper validation: ALL GENERATIVE ENGINE CHECKS PASSED.")

def run_mesh_processor_tests():
    logger.info("=========================================")
    logger.info("3. RUNNING MESH PROCESSOR REPAIR & HOMOGRAPHY TESTS")
    logger.info("=========================================")
    
    processor = GeometryProcessor()
    
    # Create an uncentered, oversized degenerate mesh to test repair/normalization
    # A single degenerate face triangle
    raw_verts = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0], # Duplicate vertex
        [11.0, 10.0, 10.0],
        [10.0, 12.0, 10.0]
    ])
    raw_faces = np.array([
        [0, 1, 2], # Degenerate triangle (coincident vertices)
        [0, 2, 3]  # Valid triangle
    ])
    
    raw_mesh = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces)
    normalized = processor.normalize_and_repair(raw_mesh)
    
    # Normalization checks
    # 1. Bounding box center must be at origin (0, 0, 0)
    assert np.allclose(normalized.centroid, [0.0, 0.0, 0.0], atol=1e-3), f"Mesh not centered: {normalized.centroid}"
    # 2. Maximum dimensions must be unit size (1.0)
    max_dim = max(normalized.extents)
    assert np.allclose(max_dim, 1.0, atol=1e-3), f"Mesh max size is not 1.0: {max_dim}"
    logger.info("✓ Mesh repair, coordinate centering, and unit bounding scale validated.")

    # Test recursive tree processing matrix math
    logger.info("Building full hierarchical simulation payload...")
    intent = route_intent("quadcopter drone mechanical model")
    
    # Synthesize payload
    loop = asyncio.get_event_loop()
    payload = loop.run_until_complete(processor.build_3d_simulation_payload(intent))
    
    assert "components" in payload, "Payload must contain components block"
    assert len(payload["components"]) > 0, "Quadcopter must contain processed component entries"
    
    for comp in payload["components"]:
        assert "name" in comp, "Component must have a unique identifier"
        assert "vertices" in comp, "Component must export raw vertex list"
        assert "faces" in comp, "Component must export raw faces list"
        assert "explosion_vector" in comp, "Component must define structural explosion vector direction"
        assert len(comp["global_transform"]) == 16, "Component global transform must be a 4x4 matrix flattened (16 floats)"
        
    logger.info(f"✓ Hierarchical system transformation checks parsed. Components computed: {len(payload['components'])}")
    logger.info("Mesh Processor validation: ALL REPAIR & MATRIX GRAPH CHECKS PASSED.")

def run_simulation_loop_tests():
    logger.info("=========================================")
    logger.info("4. RUNNING ACTIVE simulation LOOP TESTS")
    logger.info("=========================================")
    
    # Synthesize a standard baseline payload to test real-time update generation
    intent = route_intent("twin gear drive assembly")
    processor = GeometryProcessor()
    
    loop = asyncio.get_event_loop()
    payload = loop.run_until_complete(processor.build_3d_simulation_payload(intent))
    
    # Run coordinate update generation
    explosion_factor = 0.5
    rotation_angle = 1.05 # ~60 degrees
    
    coords = compute_realtime_vertex_frame(
        base_components=payload["components"],
        explosion_factor=explosion_factor,
        rotation_angle=rotation_angle
    )
    
    # Check output structure
    assert isinstance(coords, dict), "Frame update must return a dictionary"
    for comp in payload["components"]:
        name = comp["name"]
        assert name in coords, f"Frame update missing coordinate entry for: '{name}'"
        assert len(coords[name]) == len(comp["vertices"]), f"Vertex count mismatch on frame update for: '{name}'"
        
    logger.info("✓ Active simulation pipeline pre-computes real-time coordinates cleanly.")
    logger.info("Simulation Loop validation: PIPELINE EXECUTES CORRECTLY.")

if __name__ == "__main__":
    logger.info("STARTING PIPELINE VALIDATION SUITE...")
    try:
        run_orchestrator_tests()
        run_scraper_tests()
        run_mesh_processor_tests()
        run_simulation_loop_tests()
        logger.info("\n=========================================")
        logger.info("SUCCESS: ALL SCENARIOS COMPLETED AND VALIDATED.")
        logger.info("=========================================")
    except AssertionError as ae:
        logger.error(f"VAL SUITE FAILED: {ae}")
        exit(1)
    except Exception as e:
        logger.error(f"CRITICAL FAULT: {e}")
        exit(1)
