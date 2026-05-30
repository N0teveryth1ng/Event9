from __future__ import annotations
import os
import re
import math
import logging
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from live_data import (
    DataSource, BoundingBox, DataFetchPlan, LiveEntity, KNOWN_REGIONS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")


# ─────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    MECHANICAL = "mechanical"
    TERRAIN    = "terrain"
    TELEMETRY  = "telemetry"

class IntentMode(str, Enum):
    LIVE_TRACKING = "live_tracking"   # real data from internet
    ASSET_3D      = "asset_3d"        # 3D model display
    TERRAIN       = "terrain"         # geographic terrain

class PathType(str, Enum):
    GREAT_CIRCLE = "great_circle"
    GROUND       = "ground"
    ORBIT        = "orbit"

class VehicleCategory(str, Enum):
    AIRCRAFT = "aircraft"
    SHIP     = "ship"
    TRAIN    = "train"
    ROCKET   = "rocket"
    GENERIC  = "generic"


# ─────────────────────────────────────────────────────────────────
# Universal Intent — top-level output of route_intent()
# ─────────────────────────────────────────────────────────────────

class UniversalIntent(BaseModel):
    mode:          IntentMode
    display_label: str
    fetch_plan:    Optional[DataFetchPlan] = None   # live_tracking
    asset_intent:  Optional["AssetIntent"] = None  # asset_3d / terrain


# ─────────────────────────────────────────────────────────────────
# Legacy 3-D asset schemas (kept intact for mesh pipeline)
# ─────────────────────────────────────────────────────────────────

class GeoCoordinate(BaseModel):
    lat:   float
    lon:   float
    alt:   float = 0.0
    label: Optional[str] = None

class TelemetryRoute(BaseModel):
    origin:               GeoCoordinate
    destination:          GeoCoordinate
    waypoints:            List[GeoCoordinate]   = Field(default_factory=list)
    path_type:            PathType              = PathType.GREAT_CIRCLE
    cruise_altitude_m:    float                 = 10_000.0
    speed_kmh:            float                 = 900.0
    trajectory:           List[List[float]]     = Field(default_factory=list)
    total_distance_km:    float                 = 0.0
    estimated_duration_h: float                 = 0.0

class VisualTarget(BaseModel):
    canonical_name:    str
    display_name:      str
    category:          VehicleCategory = VehicleCategory.GENERIC
    vault_key:         str
    fallback_assembly: str = "aircraft"
    scale_hint:        List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])

class ComponentNode(BaseModel):
    name:              str
    primitive_type:    str             = "cube"
    relative_position: List[float]     = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    relative_rotation: List[float]     = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale:             List[float]     = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    children:          List[ComponentNode] = Field(default_factory=list)

ComponentNode.model_rebuild()

class TerrainConfig(BaseModel):
    octaves:    int   = 4
    frequency:  float = 0.05
    roughness:  float = 0.5
    min_height: float = 0.0
    max_height: float = 2.5
    sea_level:  float = 0.2

class AssetIntent(BaseModel):
    asset_name:      str
    asset_type:      AssetClass
    crawl_queries:   List[str]               = Field(default_factory=list)
    structural_tree: Optional[ComponentNode] = None
    terrain_config:  Optional[TerrainConfig] = None
    telemetry_route: Optional[TelemetryRoute] = None
    visual_target:   Optional[VisualTarget]  = None

UniversalIntent.model_rebuild()


# ─────────────────────────────────────────────────────────────────
# Keyword tables for rule-based live-tracking detection
# ─────────────────────────────────────────────────────────────────

_LIVE_SIGNALS: Dict[DataSource, List[str]] = {
    DataSource.OPENSKY_FLIGHTS: [
        "flight", "flights", "plane", "planes", "aircraft", "airline",
        "flying", "airways", "air traffic", "flightradar", "jet", "jets",
        "departing", "landing", "crossing atlantic", "crossing pacific",
        "transatlantic", "transpacific", "over the",
    ],
    DataSource.ISS_TRACKING: [
        "iss", "international space station", "space station",
    ],
    DataSource.SATELLITE_TRACK: [
        "satellite", "satellites", "orbital", "orbit", "leo", "starlink",
        "gps constellation", "sentinel", "noaa", "terra", "galileo",
    ],
    DataSource.MARITIME_AIS: [
        "ship", "ships", "vessel", "vessels", "maritime", "tanker", "tankers",
        "cargo ship", "container ship", "freighter", "fleet", "shipping lanes",
        "suez", "strait", "canal", "port", "harbor", "navy",
    ],
    DataSource.USGS_EARTHQUAKES: [
        "earthquake", "earthquakes", "seismic", "tremor", "quake",
        "magnitude", "richter", "fault",
    ],
    DataSource.OPENMETEO_WEATHER: [
        "weather", "temperature", "wind", "rain", "storm", "hurricane",
        "typhoon", "climate", "forecast", "cyclone",
    ],
    DataSource.CONFLICT_EVENTS: [
        "conflict", "war", "military", "airstrike", "strike", "battle",
        "troops", "armed", "attack", "offensive", "front line", "combat",
        "artillery", "drone strike", "missile", "insurgency", "militia",
        "casualties", "ceasefire", "ukraine", "gaza", "sudan", "yemen",
    ],
    DataSource.CYBER_THREATS: [
        "cyber", "hacker", "hack", "malware", "ransomware", "apt",
        "phishing", "ddos", "intrusion", "breach", "zero-day",
        "threat actor", "botnet", "vulnerability",
    ],
    DataSource.SIGINT_NODES: [
        "sigint", "signals intelligence", "intercept", "elint",
        "nsa", "gchq", "surveillance", "eavesdrop", "wiretap",
        "intelligence gathering", "collection", "comint",
    ],
}

_ENTITY_COLORS: Dict[DataSource, List[int]] = {
    DataSource.OPENSKY_FLIGHTS:   [0,   220, 255, 255],   # cyan
    DataSource.ISS_TRACKING:      [255, 230,   0, 255],   # yellow
    DataSource.CONFLICT_EVENTS:   [255,  61,  61, 220],   # red
    DataSource.CYBER_THREATS:     [0,   230, 118, 220],   # green
    DataSource.SIGINT_NODES:      [233,  30,  99, 220],   # magenta
    DataSource.SATELLITE_TRACK:   [100, 180, 255, 220],   # blue-white
    DataSource.MARITIME_AIS:      [0,   150, 255, 220],   # blue
    DataSource.USGS_EARTHQUAKES:  [255,  80,   0, 255],   # orange-red
    DataSource.OPENMETEO_WEATHER: [100, 180, 255, 255],   # light blue
}

_MODEL_KEYS: Dict[DataSource, str] = {
    DataSource.OPENSKY_FLIGHTS:   "generic_aircraft",
    DataSource.ISS_TRACKING:      "rocket",
    DataSource.SATELLITE_TRACK:   "satellite",
    DataSource.MARITIME_AIS:      "ship",
    DataSource.USGS_EARTHQUAKES:  "sphere",
    DataSource.OPENMETEO_WEATHER: "sphere",
    DataSource.CONFLICT_EVENTS:   "sphere",
    DataSource.CYBER_THREATS:     "sphere",
    DataSource.SIGINT_NODES:      "sphere",
}


def _detect_live_source(p: str) -> Optional[DataSource]:
    # Split into whole words so "rain" doesn't match "terrain", etc.
    words = set(re.findall(r'\b\w+\b', p))
    for source, keywords in _LIVE_SIGNALS.items():
        for kw in keywords:
            if ' ' in kw:          # multi-word phrase → substring match
                if kw in p:
                    return source
            else:                   # single word → whole-word match only
                if kw in words:
                    return source
    return None


def _detect_region(p: str) -> Optional[BoundingBox]:
    for name, bb in KNOWN_REGIONS.items():
        if name in p:
            return bb
    return None


def _detect_filter(p: str, source: DataSource) -> Optional[str]:
    """Extract airline/keyword filter from prompt."""
    for airline in ["united", "delta", "american", "british airways", "lufthansa",
                    "emirates", "qantas", "air france", "ryanair", "easyjet",
                    "singapore airlines", "cathay pacific"]:
        if airline in p:
            return airline
    return None


# ─────────────────────────────────────────────────────────────────
# Geodesic helpers (kept for 3-D asset mode)
# ─────────────────────────────────────────────────────────────────

_KNOWN_PLACES: Dict[str, Tuple[float, float]] = {
    "new york": (40.6413, -73.7781), "nyc": (40.6413, -73.7781),
    "london":   (51.4775, -0.4614),  "lhr": (51.4775, -0.4614),
    "dubai":    (25.2532, 55.3657),  "tokyo": (35.5494, 139.7798),
    "singapore":(1.3644, 103.9915),  "sydney": (-33.9461, 151.1772),
    "paris":    (48.8566, 2.3522),   "frankfurt": (50.0379, 8.5622),
    "los angeles": (33.9425, -118.4081), "la": (33.9425, -118.4081),
    "chicago":  (41.9742, -87.9073), "miami": (25.7959, -80.2870),
    "hong kong":(22.3080, 113.9185), "beijing": (40.0799, 116.6031),
    "shanghai": (31.1443, 121.8083), "mumbai": (19.0896, 72.8656),
    "delhi":    (28.5562, 77.1000),  "cairo":  (30.1219, 31.4056),
    "suez canal":(30.5852, 32.2654), "amsterdam": (52.3105, 4.7683),
    "toronto":  (43.6777, -79.6248), "seoul": (37.5665, 126.9780),
    "moscow":   (55.7558, 37.6173),  "istanbul": (41.0082, 28.9784),
    "sao paulo":(-23.5505,-46.6333), "johannesburg": (-26.2041, 28.0473),
}

def _haversine_km(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl  = math.radians(la2-la1), math.radians(lo2-lo1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _sample_great_circle(la1, lo1, la2, lo2, alt_m, n=120):
    import math
    def cart(phi, lam): return (math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi))
    def sph(x,y,z): return math.degrees(math.atan2(z,math.sqrt(x*x+y*y))), math.degrees(math.atan2(y,x))
    p1,l1 = math.radians(la1), math.radians(lo1)
    p2,l2 = math.radians(la2), math.radians(lo2)
    x1,y1,z1 = cart(p1,l1); x2,y2,z2 = cart(p2,l2)
    dot   = max(-1.0, min(1.0, x1*x2+y1*y2+z1*z2))
    omega = math.acos(dot)
    pts   = []
    for i in range(n):
        t = i/(n-1)
        if omega < 1e-10: xi,yi,zi = x1,y1,z1
        else:
            s = math.sin(omega)
            a,b = math.sin((1-t)*omega)/s, math.sin(t*omega)/s
            xi,yi,zi = a*x1+b*x2, a*y1+b*y2, a*z1+b*z2
        arc_alt = alt_m * 4*t*(1-t)
        la,lo = sph(xi,yi,zi)
        pts.append([round(la,6), round(lo,6), round(arc_alt,1)])
    return pts


# ─────────────────────────────────────────────────────────────────
# Vehicle map (for single-vehicle 3-D display)
# ─────────────────────────────────────────────────────────────────

_VEHICLE_MAP = [
    {"keywords":["boeing 777","b777","777"],"canonical":"boeing_777","display":"Boeing 777-200ER","category":VehicleCategory.AIRCRAFT,"vault_key":"boeing_777","fallback":"aircraft","scale":[64.8,19.4,60.9]},
    {"keywords":["boeing 747","b747","747","jumbo"],"canonical":"boeing_747","display":"Boeing 747-400","category":VehicleCategory.AIRCRAFT,"vault_key":"boeing_747","fallback":"aircraft","scale":[70.7,19.4,64.4]},
    {"keywords":["f22","f-22","raptor","fighter jet","stealth"],"canonical":"f22_raptor","display":"F-22 Raptor","category":VehicleCategory.AIRCRAFT,"vault_key":"f22_raptor","fallback":"fighter","scale":[18.9,5.1,13.6]},
    {"keywords":["airbus a380","a380"],"canonical":"airbus_a380","display":"Airbus A380-800","category":VehicleCategory.AIRCRAFT,"vault_key":"airbus_a380","fallback":"aircraft","scale":[79.8,24.1,79.8]},
    {"keywords":["cargo ship","container ship","freighter"],"canonical":"container_ship","display":"Panamax Container Ship","category":VehicleCategory.SHIP,"vault_key":"container_ship","fallback":"ship","scale":[294,32,24]},
    {"keywords":["cruise ship","ocean liner"],"canonical":"cruise_ship","display":"Cruise Liner","category":VehicleCategory.SHIP,"vault_key":"cruise_ship","fallback":"ship","scale":[330,40,72]},
    {"keywords":["rocket","falcon 9","starship","launch"],"canonical":"rocket","display":"Launch Vehicle","category":VehicleCategory.ROCKET,"vault_key":"rocket","fallback":"rocket","scale":[9,9,70]},
    {"keywords":["train","bullet train","shinkansen","high speed"],"canonical":"high_speed_train","display":"High-Speed Rail","category":VehicleCategory.TRAIN,"vault_key":"high_speed_train","fallback":"train","scale":[200,3.5,4]},
    {"keywords":["drone","uav","quadcopter"],"canonical":"quadcopter_uav","display":"Quadcopter UAV","category":VehicleCategory.AIRCRAFT,"vault_key":"quadcopter_uav","fallback":"drone","scale":[0.6,0.2,0.6]},
]

def _match_vehicle(p):
    for e in _VEHICLE_MAP:
        if any(kw in p for kw in e["keywords"]):
            return e
    return None

def _extract_route(p):
    matched = [(name, coords) for name, coords in _KNOWN_PLACES.items() if name in p]
    if len(matched) < 2:
        return None
    matched.sort(key=lambda x: p.index(x[0]))
    n1,c1 = matched[0]; n2,c2 = matched[1]
    o = GeoCoordinate(lat=c1[0], lon=c1[1], label=n1.title())
    d = GeoCoordinate(lat=c2[0], lon=c2[1], label=n2.title())
    dist = _haversine_km(o.lat, o.lon, d.lat, d.lon)
    is_ship = any(kw in p for kw in ["ship","vessel","tanker","freighter"])
    spd = 35.0 if is_ship else 900.0
    alt = 0.0  if is_ship else 10_000.0
    traj = _sample_great_circle(o.lat, o.lon, d.lat, d.lon, alt)
    return TelemetryRoute(
        origin=o, destination=d,
        path_type=PathType.GREAT_CIRCLE,
        cruise_altitude_m=alt, speed_kmh=spd,
        trajectory=traj,
        total_distance_km=round(dist,1),
        estimated_duration_h=round(dist/max(spd,1),2),
    )


# ─────────────────────────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────────────────────────

def get_gemini_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.warning("GEMINI_API_KEY not set — using rule-based fallback.")
        return None
    try:
        from google import genai
        return genai.Client()
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Rule-based fallback
# ─────────────────────────────────────────────────────────────────

def _rule_based_intent(prompt: str) -> UniversalIntent:
    p = prompt.lower().strip()

    # 1 ── Live tracking detection (highest priority) ──────────────
    source = _detect_live_source(p)
    if source:
        bb     = _detect_region(p)
        filt   = _detect_filter(p, source)
        label  = prompt.strip()

        # For flights without explicit region, default to global
        if source == DataSource.OPENSKY_FLIGHTS and bb is None:
            for place, (lat, lon) in _KNOWN_PLACES.items():
                if place in p:
                    pad = 20
                    bb = BoundingBox(
                        lat_min=lat-pad, lat_max=lat+pad,
                        lon_min=lon-pad, lon_max=lon+pad
                    )
                    break
            if bb is None:
                bb = KNOWN_REGIONS["global"]

        # Weather: pass named location as location_query for geocoding
        loc_q = None
        if source == DataSource.OPENMETEO_WEATHER:
            for place in _KNOWN_PLACES:
                if place in p:
                    loc_q = place
                    break

        # Conflict: pass original prompt as filter keyword for GDELT query
        conflict_kw = filt
        if source == DataSource.CONFLICT_EVENTS:
            conflict_kw = p[:80]

        plan = DataFetchPlan(
            source          = source,
            display_label   = label,
            model_key       = _MODEL_KEYS.get(source, "generic_aircraft"),
            entity_color    = _ENTITY_COLORS.get(source, [0,220,255,255]),
            bbox            = bb,
            filter_keyword  = conflict_kw if source == DataSource.CONFLICT_EVENTS else filt,
            location_query  = loc_q,
            max_entities    = 500 if source == DataSource.OPENSKY_FLIGHTS else 200,
            refresh_s       = 15.0 if source == DataSource.OPENSKY_FLIGHTS else 10.0,
        )
        return UniversalIntent(mode=IntentMode.LIVE_TRACKING, display_label=label, fetch_plan=plan)

    # 2 ── Terrain ─────────────────────────────────────────────────
    if any(kw in p for kw in ["terrain","mountain","valley","island","landscape","elevation","canyon"]):
        ai = AssetIntent(
            asset_name  = prompt.strip(),
            asset_type  = AssetClass.TERRAIN,
            crawl_queries = [f"{prompt} dem heightmap"],
            terrain_config = TerrainConfig(
                octaves   = 5 if "mountain" in p else 3,
                frequency = 0.03 if "valley" in p else 0.06,
                roughness = 0.65 if "mountain" in p else 0.4,
                min_height= 0.0,
                max_height= 3.5 if "mountain" in p else 1.5,
                sea_level = 0.15,
            )
        )
        return UniversalIntent(mode=IntentMode.TERRAIN, display_label=prompt.strip(), asset_intent=ai)

    # 3 ── Single-vehicle route (3-D model on telemetry arc) ───────
    veh   = _match_vehicle(p)
    route = _extract_route(p)
    if veh and route:
        vt = VisualTarget(
            canonical_name=veh["canonical"], display_name=veh["display"],
            category=veh["category"], vault_key=veh["vault_key"],
            fallback_assembly=veh["fallback"], scale_hint=veh["scale"],
        )
        ai = AssetIntent(
            asset_name     = f"{veh['display']} — {route.origin.label} → {route.destination.label}",
            asset_type     = AssetClass.TELEMETRY,
            crawl_queries  = [f"{veh['display']} 3D model OBJ"],
            telemetry_route= route,
            visual_target  = vt,
        )
        return UniversalIntent(mode=IntentMode.ASSET_3D,
                               display_label=ai.asset_name, asset_intent=ai)

    # 4 ── Mechanical 3-D model ────────────────────────────────────
    base = _build_mechanical_tree(p)
    ai   = AssetIntent(
        asset_name    = prompt.strip() or "Mechanical Assembly",
        asset_type    = AssetClass.MECHANICAL,
        crawl_queries = [f"{prompt} CAD stl", f"{prompt} mechanical model"],
        structural_tree = base,
    )
    return UniversalIntent(mode=IntentMode.ASSET_3D,
                           display_label=ai.asset_name, asset_intent=ai)


def _build_mechanical_tree(p: str) -> ComponentNode:
    if any(kw in p for kw in ["robot","arm","manipulator"]):
        return ComponentNode(name="robot_base", primitive_type="cylinder", scale=[1.2,0.3,1.2],
            children=[ComponentNode(name="lower_arm", primitive_type="shaft", relative_position=[0,0.8,0],
                relative_rotation=[0,0.2,0], scale=[0.3,1.5,0.3],
                children=[ComponentNode(name="upper_arm", primitive_type="shaft",
                    relative_position=[0,1.5,0], relative_rotation=[0,-0.4,0], scale=[0.25,1.2,0.25],
                    children=[ComponentNode(name="end_effector", primitive_type="gear",
                        relative_position=[0,1.2,0], scale=[0.6,0.15,0.6])])])])
    if any(kw in p for kw in ["plane","airplane","jet","aircraft","flight","glider"]):
        return ComponentNode(name="fuselage", primitive_type="cylinder", scale=[0.4,0.4,3.6],
            children=[
                ComponentNode(name="left_wing",  primitive_type="box", relative_position=[-1.6,0,0.2], relative_rotation=[0,0.08,-0.05], scale=[1.6,0.04,0.6]),
                ComponentNode(name="right_wing", primitive_type="box", relative_position=[1.6,0,0.2],  relative_rotation=[0,-0.08,0.05], scale=[1.6,0.04,0.6]),
                ComponentNode(name="vstab",      primitive_type="box", relative_position=[0,0.45,-1.4], scale=[0.05,0.6,0.35]),
                ComponentNode(name="engine_L",   primitive_type="cylinder", relative_position=[-0.7,-0.15,0.2], scale=[0.22,0.22,0.55]),
                ComponentNode(name="engine_R",   primitive_type="cylinder", relative_position=[0.7,-0.15,0.2],  scale=[0.22,0.22,0.55]),
            ])
    if any(kw in p for kw in ["drone","uav","quadcopter"]):
        arms = []
        for name,pos,rot in [("arm_fl",[-0.5,0,0.5],[1.57,0,0.785]),("arm_fr",[0.5,0,0.5],[1.57,0,-0.785]),
                              ("arm_bl",[-0.5,0,-0.5],[1.57,0,-0.785]),("arm_br",[0.5,0,-0.5],[1.57,0,0.785])]:
            arms.append(ComponentNode(name=name, primitive_type="shaft", relative_position=pos,
                relative_rotation=rot, scale=[0.08,0.7,0.08],
                children=[ComponentNode(name=f"rotor_{name[-2:]}", primitive_type="gear",
                    relative_position=[0,0.35,0], scale=[0.45,0.02,0.45])]))
        return ComponentNode(name="uav_hub", primitive_type="cylinder", scale=[0.7,0.15,0.7], children=arms)
    if any(kw in p for kw in ["car","vehicle","truck","automobile"]):
        return ComponentNode(name="chassis", primitive_type="box", scale=[1.0,0.3,2.2],
            children=[
                ComponentNode(name="wheel_fl", primitive_type="cylinder", relative_position=[-0.55,-0.15,0.75], relative_rotation=[0,0,1.57], scale=[0.35,0.15,0.35]),
                ComponentNode(name="wheel_fr", primitive_type="cylinder", relative_position=[0.55,-0.15,0.75],  relative_rotation=[0,0,1.57], scale=[0.35,0.15,0.35]),
                ComponentNode(name="wheel_bl", primitive_type="cylinder", relative_position=[-0.55,-0.15,-0.75], relative_rotation=[0,0,1.57], scale=[0.35,0.15,0.35]),
                ComponentNode(name="wheel_br", primitive_type="cylinder", relative_position=[0.55,-0.15,-0.75],  relative_rotation=[0,0,1.57], scale=[0.35,0.15,0.35]),
                ComponentNode(name="cabin",    primitive_type="box", relative_position=[0,0.3,-0.15], scale=[0.8,0.4,1.1]),
            ])
    if any(kw in p for kw in ["engine","motor","piston","turbine","crankshaft"]):
        return ComponentNode(name="housing", primitive_type="box", scale=[1.2,0.8,1.8],
            children=[
                ComponentNode(name="crankshaft", primitive_type="shaft", relative_position=[0,0,0], scale=[0.15,0.15,1.6],
                    children=[
                        ComponentNode(name="piston_1", primitive_type="cylinder", relative_position=[0.3,0.3,0.5], scale=[0.2,0.2,0.4]),
                        ComponentNode(name="piston_2", primitive_type="cylinder", relative_position=[-0.3,0.3,0],   scale=[0.2,0.2,0.4]),
                        ComponentNode(name="piston_3", primitive_type="cylinder", relative_position=[0.3,0.3,-0.5], scale=[0.2,0.2,0.4]),
                        ComponentNode(name="flywheel", primitive_type="cylinder", relative_position=[0,-0.1,-0.85], scale=[0.6,0.6,0.1]),
                    ]),
                ComponentNode(name="gear_train", primitive_type="gear", relative_position=[0,0.5,0.6], scale=[0.5,0.1,0.5]),
            ])
    # Generic mechanical
    return ComponentNode(name="drive_shaft", primitive_type="shaft", scale=[0.2,2.0,0.2],
        children=[
            ComponentNode(name="spur_gear", primitive_type="gear", relative_position=[0,0.5,0], relative_rotation=[1.57,0,0], scale=[1.0,0.2,1.0]),
            ComponentNode(name="flywheel",  primitive_type="cylinder", relative_position=[0,-0.6,0], scale=[1.5,0.15,1.5],
                children=[ComponentNode(name="crank_pin", primitive_type="cylinder", relative_position=[0.6,0.2,0], scale=[0.1,0.4,0.1])]),
        ])


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def route_intent(prompt: str) -> UniversalIntent:
    """
    Routes any natural-language prompt to a UniversalIntent.
    Tries Gemini first; falls back to rule-based classification.
    """
    client = get_gemini_client()
    if client:
        try:
            return _gemini_route(client, prompt)
        except Exception as e:
            logger.error(f"Gemini routing failed: {e} — using fallback")

    return _rule_based_intent(prompt)


def _gemini_route(client, prompt: str) -> UniversalIntent:
    """
    Ask Gemini to classify the prompt into a compact routing decision,
    then map it back to a full UniversalIntent via the rule-based helpers.
    We avoid passing the recursive ComponentNode schema to Gemini.
    """
    from google.genai import types
    import json

    # Minimal flat schema — no recursion
    class _GeminiDecision(BaseModel):
        mode:          str   # "live_tracking" | "asset_3d" | "terrain"
        source:        Optional[str] = None   # DataSource value for live_tracking
        display_label: str  = ""
        lat_min:       Optional[float] = None
        lat_max:       Optional[float] = None
        lon_min:       Optional[float] = None
        lon_max:       Optional[float] = None
        filter_keyword:Optional[str]  = None

    system = """You are the routing brain of EventNine — a global situational awareness dashboard.

Classify every query. Return JSON with these fields:
  mode          : "live_tracking" | "asset_3d" | "terrain"
  source        : (only for live_tracking) one of:
                  "opensky_flights"   — flights, planes, aircraft, air traffic
                  "iss_tracking"      — ISS, space station, satellite, orbit
                  "usgs_earthquakes"  — earthquakes, seismic, tremors, quakes
                  "openmeteo_weather" — weather, temperature, storm, wind
  display_label : short human-readable title for the query
  lat_min/lat_max/lon_min/lon_max : bounding box (omit for global or non-live)
  filter_keyword: airline name / callsign filter if specified (else omit)

Examples:
  "flights over North America" -> mode=live_tracking, source=opensky_flights, lat_min=15,lat_max=72,lon_min=-170,lon_max=-50
  "ISS position now"           -> mode=live_tracking, source=iss_tracking
  "earthquakes today"          -> mode=live_tracking, source=usgs_earthquakes
  "show me a car engine"       -> mode=asset_3d
  "Boeing 777 NYC to London"   -> mode=asset_3d
  "mountain terrain"           -> mode=terrain"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_GeminiDecision,
            system_instruction=system,
            temperature=0.05,
        ),
    )

    d = _GeminiDecision.model_validate_json(response.text)
    logger.info(f"Gemini decision: mode={d.mode} source={d.source} label={d.display_label!r}")

    label = d.display_label or prompt.strip()

    if d.mode == "live_tracking" and d.source:
        try:
            src = DataSource(d.source)
        except ValueError:
            src = DataSource.GENERIC_ROUTE

        bb = None
        if all(x is not None for x in [d.lat_min, d.lat_max, d.lon_min, d.lon_max]):
            bb = BoundingBox(lat_min=d.lat_min, lat_max=d.lat_max,
                             lon_min=d.lon_min, lon_max=d.lon_max)
        elif src == DataSource.OPENSKY_FLIGHTS:
            bb = KNOWN_REGIONS["global"]

        plan = DataFetchPlan(
            source        = src,
            display_label = label,
            model_key     = _MODEL_KEYS.get(src, "generic_aircraft"),
            entity_color  = _ENTITY_COLORS.get(src, [0, 220, 255, 255]),
            bbox          = bb,
            filter_keyword= d.filter_keyword,
            max_entities  = 500 if src == DataSource.OPENSKY_FLIGHTS else 200,
            refresh_s     = 15.0 if src == DataSource.OPENSKY_FLIGHTS else 10.0,
        )
        return UniversalIntent(mode=IntentMode.LIVE_TRACKING, display_label=label, fetch_plan=plan)

    # asset_3d or terrain — let rule-based build the full intent
    return _rule_based_intent(prompt)


# ═════════════════════════════════════════════════════════════════
# 9-DOMAIN OPERATIONAL INTELLIGENCE SCHEMAS
# ═════════════════════════════════════════════════════════════════

class TargetNode(BaseModel):
    node_id:           str
    designation:       str
    domain:            str   # space|maritime|threat|weather|cyber|sigint|commodity|trade|geo
    velocity_knots:    float
    heading:           float
    current_position:  Dict[str, float]          # {"lat": float, "lng": float}
    trajectory_path:   List[Dict[str, float]]    # 120-point geodesic arc
    tactical_metadata: Dict[str, Any]

class ThreatIntelligencePayload(BaseModel):
    operational_theater: str
    bounding_box:        List[float]   # [min_lat, min_lon, max_lat, max_lon]
    threat_index_level:  str           # ALPHA_CLEAR | BRAVO_MONITORED | OMEGA_CRITICAL
    system_integrity:    float
    interlinked_nodes:   List[TargetNode]


# ─────────────────────────────────────────────────────────────────
# NLP KEYWORD ENGINE — domain + threat classification
# ─────────────────────────────────────────────────────────────────

_AGGRESSIVE_TERMS = {
    "war", "conflict", "intercept", "strike", "attack", "hostile",
    "missile", "threat", "combat", "military", "invasion", "siege",
    "offensive", "airspace", "patrol", "intercept", "engagement",
    "battalion", "brigade", "squadron", "carrier", "submarine",
}
_CYBER_TERMS = {
    "cyber", "infrastructure", "hack", "breach", "grid", "cable",
    "network", "ddos", "malware", "ransomware", "darknet", "exploit",
}
_SIGINT_TERMS = {
    "sigint", "signal", "frequency", "surveillance", "radar",
    "comint", "elint", "intercept", "humint", "intel",
}
_WEATHER_TERMS = {
    "storm", "hurricane", "typhoon", "cyclone", "weather", "atmospheric",
    "pressure", "wind", "rainfall", "flood",
}
_SPACE_TERMS = {
    "space", "satellite", "orbit", "leo", "iss", "starlink", "orbital",
    "constellation", "gps", "sar", "recon",
}
_MARITIME_TERMS = {
    "maritime", "ship", "vessel", "fleet", "navy", "port", "chokepoint",
    "strait", "canal", "tanker", "carrier",
}
_COMMODITY_TERMS = {
    "commodity", "oil", "crude", "lng", "gold", "copper", "wheat",
    "iron", "gas", "energy", "pipeline",
}
_TRADE_TERMS = {
    "trade", "cargo", "container", "shipping", "route", "corridor",
    "export", "import", "supply chain",
}
_THREAT_TERMS = {
    "threat", "war", "conflict", "military", "combat", "missile",
    "strike", "defense", "hostile",
}

_DOMAIN_KEYWORD_MAP: Dict[str, set] = {
    "space":     _SPACE_TERMS,
    "maritime":  _MARITIME_TERMS,
    "threat":    _THREAT_TERMS,
    "weather":   _WEATHER_TERMS,
    "cyber":     _CYBER_TERMS,
    "sigint":    _SIGINT_TERMS,
    "commodity": _COMMODITY_TERMS,
    "trade":     _TRADE_TERMS,
}

_THEATER_KEYWORDS: Dict[str, str] = {
    "europe":        "European Theater",
    "asia":          "Indo-Pacific Theater",
    "pacific":       "Indo-Pacific Theater",
    "atlantic":      "Atlantic Theater",
    "middle east":   "Middle East Theater",
    "africa":        "African Theater",
    "arctic":        "Arctic Theater",
    "global":        "Global Operations",
    "north america": "North American Theater",
    "south china sea": "South China Sea Theater",
    "eastern europe": "Eastern European Theater",
}


def parse_operational_intent(prompt: str) -> Dict[str, Any]:
    """
    NLP keyword engine → returns routing dict for the ops-stream WebSocket.
    Forces OMEGA_CRITICAL on aggressive terms; degrades integrity on cyber/sigint.
    """
    p     = prompt.lower().strip()
    words = set(re.findall(r'\b\w+\b', p))

    # ── Threat level ─────────────────────────────────────────────
    if words & _AGGRESSIVE_TERMS:
        threat_level = "OMEGA_CRITICAL"
        integrity    = round(random.uniform(0.45, 0.72), 3)
    elif words & (_CYBER_TERMS | _SIGINT_TERMS):
        threat_level = "BRAVO_MONITORED"
        integrity    = round(random.uniform(0.68, 0.87), 3)
    else:
        threat_level = "ALPHA_CLEAR"
        integrity    = round(random.uniform(0.88, 0.99), 3)

    # ── Domain detection (multi-domain supported) ─────────────────
    detected_domains: List[str] = []
    for domain, kw_set in _DOMAIN_KEYWORD_MAP.items():
        if words & kw_set:
            detected_domains.append(domain)

    # Specific phrase checks
    if "cyber" in p or "cable" in p:
        if "cyber" not in detected_domains:
            detected_domains.append("cyber")
    if not detected_domains:
        # Default: show all domains on generic queries
        detected_domains = ["space", "maritime", "trade", "weather", "commodity"]

    # ── Theater ───────────────────────────────────────────────────
    theater = "Global Operations"
    for kw, label in _THEATER_KEYWORDS.items():
        if kw in p:
            theater = label
            break

    # ── Bounding box ──────────────────────────────────────────────
    bbox_map = {
        "European Theater":        [36.0, -10.0, 71.0, 42.0],
        "Indo-Pacific Theater":    [-10.0, 60.0, 55.0, 180.0],
        "Atlantic Theater":        [15.0, -65.0, 65.0, -5.0],
        "Middle East Theater":     [15.0, 33.0, 42.0, 65.0],
        "African Theater":         [-35.0, -18.0, 37.0, 52.0],
        "Arctic Theater":          [66.0, -180.0, 90.0, 180.0],
        "North American Theater":  [15.0, -170.0, 72.0, -50.0],
        "South China Sea Theater": [0.0, 105.0, 25.0, 125.0],
        "Eastern European Theater":[44.0, 22.0, 58.0, 45.0],
        "Global Operations":       [-90.0, -180.0, 90.0, 180.0],
    }
    bbox = bbox_map.get(theater, [-90.0, -180.0, 90.0, 180.0])

    return {
        "theater":        theater,
        "domains":        detected_domains,
        "threat_level":   threat_level,
        "bbox":           bbox,
        "integrity":      integrity,
        "raw_prompt":     prompt,
    }


import random  # used by parse_operational_intent — placed here to avoid circular
