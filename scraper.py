"""
scraper.py — Stateless Geodesic Intelligence Pipeline
======================================================
9-Domain operational intelligence engine. Zero disk reads/writes.
All trimesh / vertex / polygon code removed.
Pure NumPy great-circle geodesics + domain fleet simulation.
"""
import math
import uuid
import time
import random
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger("GeodesicEngine")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGIC SECTORS — chokepoints, cables, orbital tracks, conflict zones
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIC_SECTORS: Dict[str, Any] = {
    "maritime_chokepoints": [
        {"name": "Strait of Malacca",       "lat":  2.50, "lng": 103.00, "traffic_density": 0.92},
        {"name": "Suez Canal",               "lat": 30.58, "lng":  32.27, "traffic_density": 0.88},
        {"name": "Strait of Hormuz",         "lat": 26.60, "lng":  56.30, "traffic_density": 0.85},
        {"name": "Panama Canal",             "lat":  9.10, "lng": -79.68, "traffic_density": 0.80},
        {"name": "Bab-el-Mandeb",            "lat": 12.58, "lng":  43.42, "traffic_density": 0.75},
        {"name": "English Channel",          "lat": 51.00, "lng":   1.50, "traffic_density": 0.90},
        {"name": "Taiwan Strait",            "lat": 24.50, "lng": 119.50, "traffic_density": 0.78},
        {"name": "Danish Straits",           "lat": 57.00, "lng":  10.50, "traffic_density": 0.65},
        {"name": "Strait of Gibraltar",      "lat": 35.98, "lng":  -5.48, "traffic_density": 0.82},
        {"name": "Cape of Good Hope",        "lat":-34.36, "lng":  18.47, "traffic_density": 0.60},
    ],
    "cyber_infrastructure": [
        {"name": "Trans-Atlantic Cable 1",   "origin": (40.71, -74.00), "dest": (51.51,  -0.13)},
        {"name": "Trans-Atlantic Cable 2",   "origin": (25.77, -80.19), "dest": (48.85,   2.35)},
        {"name": "Trans-Pacific Cable",      "origin": (37.77,-122.42), "dest": (35.68, 139.69)},
        {"name": "SEAMEWE-5 Cable",          "origin": (1.352, 103.82), "dest": (48.85,   2.35)},
        {"name": "Africa-Europe Cable",      "origin": (6.52,   3.38),  "dest": (51.51,  -0.13)},
        {"name": "India-ME-WEurope Cable",   "origin": (19.07,  72.88), "dest": (29.98,  31.13)},
        {"name": "Pacific Light Cable",      "origin": (22.30, 114.19), "dest": (37.77,-122.42)},
        {"name": "Southern Cross Cable",     "origin":(-33.87, 151.21), "dest": (37.77,-122.42)},
    ],
    "leo_orbital_tracks": [
        {"name": "ISS Track",                "altitude_km": 408,  "inclination": 51.6, "period_min": 92.9},
        {"name": "Starlink Shell-1",         "altitude_km": 550,  "inclination": 53.0, "period_min": 95.5},
        {"name": "Starlink Shell-2",         "altitude_km": 540,  "inclination": 53.2, "period_min": 95.1},
        {"name": "OneWeb Constellation",     "altitude_km": 1200, "inclination": 87.9, "period_min": 109.3},
        {"name": "GPS IIF Orbit",            "altitude_km": 20200,"inclination": 55.0, "period_min": 718.0},
        {"name": "Iridium NEXT",             "altitude_km": 780,  "inclination": 86.4, "period_min": 100.4},
    ],
    "conflict_zones": [
        {"name": "Eastern Europe Grid",      "lat": 50.00, "lng":  30.00, "radius_km": 800,  "intensity": 0.95},
        {"name": "South China Sea",          "lat": 15.00, "lng": 114.00, "radius_km": 900,  "intensity": 0.80},
        {"name": "Middle East Theater",      "lat": 33.00, "lng":  44.00, "radius_km": 700,  "intensity": 0.88},
        {"name": "Korean Peninsula",         "lat": 38.00, "lng": 127.00, "radius_km": 300,  "intensity": 0.70},
        {"name": "Sahel Region",             "lat": 14.00, "lng":   2.00, "radius_km": 600,  "intensity": 0.65},
        {"name": "Taiwan Strait Zone",       "lat": 24.00, "lng": 120.00, "radius_km": 350,  "intensity": 0.82},
    ],
    "trade_corridors": [
        {"name": "Asia-Europe Main Line",    "origin": (1.35, 103.82),  "dest": (51.51,  -0.13)},
        {"name": "Transpacific Eastbound",   "origin": (22.30, 114.19), "dest": (33.75,-118.19)},
        {"name": "Transatlantic Route",      "origin": (51.51,  -0.13), "dest": (40.71, -74.00)},
        {"name": "US Gulf-Europe",           "origin": (29.74, -95.37), "dest": (51.51,  -0.13)},
        {"name": "Persian Gulf-Asia",        "origin": (26.20,  50.65), "dest": (22.30, 114.19)},
        {"name": "West Africa-Europe",       "origin": ( 6.52,   3.38), "dest": (51.51,  -0.13)},
        {"name": "South America-Asia",       "origin": (-23.55,-46.63), "dest": (31.23, 121.47)},
        {"name": "Australia-China",          "origin": (-33.87, 151.21),"dest": (31.23, 121.47)},
    ],
    "sigint_stations": [
        {"name": "GCHQ Bude",                "lat": 50.89, "lng":  -4.55, "freq_mhz": 14200, "range_km": 3000},
        {"name": "NSA Fort Meade",           "lat": 39.11, "lng": -76.77, "freq_mhz": 18500, "range_km": 4000},
        {"name": "Pine Gap",                 "lat":-23.80, "lng": 133.74, "freq_mhz": 22000, "range_km": 3500},
        {"name": "Menwith Hill",             "lat": 54.00, "lng":  -1.69, "freq_mhz": 16800, "range_km": 3200},
        {"name": "Diego Garcia",             "lat": -7.31, "lng":  72.41, "freq_mhz": 12400, "range_km": 4500},
        {"name": "Misawa",                   "lat": 40.70, "lng": 141.37, "freq_mhz": 19600, "range_km": 2800},
    ],
    "commodity_flows": [
        {"name": "Brent Crude — North Sea",  "origin": (61.00,   2.00),  "dest": (51.51,  -0.13), "commodity": "crude_oil",    "volume_kbd": 1800},
        {"name": "WTI Crude — Cushing OK",   "origin": (35.98, -96.77),  "dest": (29.74, -95.37), "commodity": "crude_oil",    "volume_kbd": 2200},
        {"name": "LNG — Qatar to Japan",     "origin": (25.29,  51.53),  "dest": (35.68, 139.69), "commodity": "lng",          "volume_kbd": 600},
        {"name": "Iron Ore — Pilbara",       "origin": (-23.36, 119.74), "dest": (31.23, 121.47), "commodity": "iron_ore",     "volume_kbd": 850},
        {"name": "Wheat — Black Sea",        "origin": (46.48,  30.73),  "dest": (36.89,  30.70), "commodity": "wheat",        "volume_kbd": 320},
        {"name": "Gold — Johannesburg",      "origin": (-26.20,  28.05), "dest": (51.51,  -0.13), "commodity": "gold",         "volume_kbd": 45},
        {"name": "Copper — Chile",           "origin": (-33.45, -70.67), "dest": (31.23, 121.47), "commodity": "copper",       "volume_kbd": 280},
        {"name": "LNG — Australia to Korea", "origin": (-33.87, 151.21), "dest": (37.56, 126.97), "commodity": "lng",          "volume_kbd": 420},
    ],
    "weather_systems": [
        {"name": "Western Pacific Typhoon Belt", "lat":  18.0, "lng": 135.0, "radius_km": 1200, "category": 4},
        {"name": "Atlantic Hurricane Corridor",  "lat":  24.0, "lng": -65.0, "radius_km": 900,  "category": 3},
        {"name": "Indian Ocean Cyclone Basin",   "lat": -15.0, "lng":  75.0, "radius_km": 800,  "category": 3},
        {"name": "Bay of Bengal Storm System",   "lat":  14.0, "lng":  88.0, "radius_km": 700,  "category": 2},
        {"name": "Arabian Sea Cyclone",          "lat":  16.0, "lng":  62.0, "radius_km": 600,  "category": 2},
        {"name": "Southern Ocean Polar Low",     "lat": -58.0, "lng":  10.0, "radius_km": 500,  "category": 1},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# GEODESIC TRAJECTORY GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_geodesic_trajectory(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    steps: int = 120,
) -> List[Dict[str, float]]:
    """
    Spherical law of cosines great-circle interpolation.
    Returns a list of {lat, lng} dicts along the shortest arc.
    """
    lat1 = math.radians(origin[0])
    lon1 = math.radians(origin[1])
    lat2 = math.radians(destination[0])
    lon2 = math.radians(destination[1])

    cos_d = (math.sin(lat1) * math.sin(lat2)
             + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
    cos_d = max(-1.0, min(1.0, cos_d))
    d = math.acos(cos_d)

    points: List[Dict[str, float]] = []
    if d < 1e-10:
        return [{"lat": origin[0], "lng": origin[1]}] * steps

    sin_d = math.sin(d)
    for i in range(steps):
        t = i / (steps - 1)
        A = math.sin((1 - t) * d) / sin_d
        B = math.sin(t * d) / sin_d
        x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
        y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
        z = A * math.sin(lat1)                  + B * math.sin(lat2)
        lat_i = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
        lon_i = math.degrees(math.atan2(y, x))
        points.append({"lat": round(lat_i, 5), "lng": round(lon_i, 5)})

    return points


def _geodesic_heading(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Initial bearing from p1 to p2 (degrees, 0=North clockwise)."""
    lat1 = math.radians(p1["lat"]); lat2 = math.radians(p2["lat"])
    dlon = math.radians(p2["lng"] - p1["lng"])
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN COLORS
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_COLORS: Dict[str, List[int]] = {
    "space":     [180, 120, 255, 220],   # violet
    "maritime":  [0,   180, 255, 220],   # cyan-blue
    "threat":    [255,  40,  40, 220],   # red
    "weather":   [255, 200,  50, 220],   # amber
    "cyber":     [50,  255, 150, 220],   # green
    "sigint":    [255, 140, 255, 220],   # magenta
    "commodity": [255, 160,  50, 220],   # orange
    "trade":     [100, 200, 255, 220],   # sky
    "geo":       [160, 160, 180, 220],   # slate
}

DOMAIN_SPEEDS: Dict[str, Tuple[float, float]] = {
    "space":     (15000, 28000),   # knots (LEO: ~17000 knots = 28000 km/h / 1.852)
    "maritime":  (8,     22),      # knots
    "threat":    (400,   1200),    # knots (aircraft/missiles)
    "weather":   (5,     30),      # knots
    "cyber":     (50000, 80000),   # knots (light speed equivalent symbolic)
    "sigint":    (0,     0),       # stationary nodes
    "commodity": (10,    20),      # knots (tankers)
    "trade":     (12,    24),      # knots
    "geo":       (0,     0),       # static
}


# ─────────────────────────────────────────────────────────────────────────────
# FLEET SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class FleetSimulator:
    """
    Stateless 9-domain fleet generator.
    Positions advance along pre-computed geodesic arcs on every tick.
    """

    def __init__(self):
        self._fleets:  Dict[str, List[Dict[str, Any]]] = {}  # domain → nodes
        self._theater: str  = "Global Operations"
        self._threat:  str  = "ALPHA_CLEAR"
        self._bbox:    List[float] = [-90, -180, 90, 180]
        self._integrity: float = 1.0
        self._active: bool = False

    # ── Public activation ─────────────────────────────────────────────────

    def activate(self, theater: str, domains: List[str],
                 threat_level: str, bbox: List[float],
                 integrity: float = 1.0):
        self._theater  = theater
        self._threat   = threat_level
        self._bbox     = bbox
        self._integrity = integrity
        self._fleets   = {}
        self._active   = True
        for domain in domains:
            self._fleets[domain] = self._generate_domain_fleet(domain, bbox)

    def deactivate(self):
        self._active = False
        self._fleets = {}

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Tick — advance all nodes along their trajectories ─────────────────

    def tick(self) -> Dict[str, Any]:
        """
        Advances every node by one step and returns a serialisable payload.
        Called at 30 Hz from the WebSocket broadcast loop.
        """
        nodes = []
        for domain, fleet in self._fleets.items():
            for node in fleet:
                self._advance_node(node)
                nodes.append({
                    "node_id":          node["node_id"],
                    "designation":      node["designation"],
                    "domain":           domain,
                    "velocity_knots":   round(node["velocity_knots"], 1),
                    "heading":          round(node["heading"], 1),
                    "current_position": node["current_position"],
                    "trajectory_path":  node["trajectory_path"],
                    "tactical_metadata": node["tactical_metadata"],
                    "color":            DOMAIN_COLORS.get(domain, [200, 200, 200, 220]),
                })
        return {
            "operational_theater": self._theater,
            "bounding_box":        self._bbox,
            "threat_index_level":  self._threat,
            "system_integrity":    round(self._integrity, 3),
            "interlinked_nodes":   nodes,
            "timestamp":           time.time(),
        }

    # ── Node advancement ──────────────────────────────────────────────────

    def _advance_node(self, node: Dict[str, Any]):
        path = node["trajectory_path"]
        if not path or node["velocity_knots"] < 0.1:
            return
        idx  = node["_path_index"]
        idx  = (idx + 1) % len(path)
        node["_path_index"]     = idx
        node["current_position"] = path[idx]

        nxt = path[(idx + 1) % len(path)]
        node["heading"] = _geodesic_heading(path[idx], nxt)

        # Degrade system integrity slightly under OMEGA_CRITICAL
        if self._threat == "OMEGA_CRITICAL":
            node["tactical_metadata"]["signal_strength"] = round(
                max(0.1, node["tactical_metadata"].get("signal_strength", 1.0)
                    - random.uniform(0, 0.002)), 3)

    # ── Domain fleet builders ─────────────────────────────────────────────

    def _generate_domain_fleet(self, domain: str,
                                bbox: List[float]) -> List[Dict[str, Any]]:
        try:
            builders = {
                "space":     self._build_space_fleet,
                "maritime":  self._build_maritime_fleet,
                "threat":    self._build_threat_fleet,
                "weather":   self._build_weather_fleet,
                "cyber":     self._build_cyber_fleet,
                "sigint":    self._build_sigint_fleet,
                "commodity": self._build_commodity_fleet,
                "trade":     self._build_trade_fleet,
                "geo":       self._build_geo_fleet,
            }
            fn = builders.get(domain, self._build_generic_fleet)
            return fn(bbox)
        except Exception as e:
            logger.error(f"Fleet build failed for domain={domain}: {e}")
            return self._build_generic_fleet(bbox)

    # ── Space ─────────────────────────────────────────────────────────────

    def _build_space_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        for track in STRATEGIC_SECTORS["leo_orbital_tracks"]:
            inclination = track["inclination"]
            alt_km      = track["altitude_km"]
            # Synthesise 3 satellites per orbital track at different phases
            for phase_offset in [0, 40, 80]:
                # Great circle at this inclination: origin → destination
                # Approximate as equatorial-tilted arc
                start_lon = (phase_offset * 3) % 360 - 180
                origin = (0.0, start_lon)
                dest   = (inclination * math.cos(math.radians(start_lon + 90)),
                          (start_lon + 180) % 360 - 180)
                traj = calculate_geodesic_trajectory(origin, dest, 120)
                spd  = round(random.uniform(14800, 15200), 1)
                node = self._make_node(
                    f"{track['name']} SV-{phase_offset//40+1}",
                    "space", traj, spd, phase_offset * 2,
                    {
                        "altitude_km":   alt_km,
                        "inclination":   inclination,
                        "period_min":    track["period_min"],
                        "signal_strength": round(random.uniform(0.88, 0.99), 3),
                        "orbital_type":  "LEO" if alt_km < 2000 else "MEO",
                    }
                )
                fleet.append(node)
        return fleet

    # ── Maritime ──────────────────────────────────────────────────────────

    def _build_maritime_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        chokepoints = STRATEGIC_SECTORS["maritime_chokepoints"]
        for i, cp in enumerate(chokepoints[:8]):
            # Pair each chokepoint with the next as origin→dest
            dest_cp = chokepoints[(i + 3) % len(chokepoints)]
            origin  = (cp["lat"],      cp["lng"])
            dest    = (dest_cp["lat"], dest_cp["lng"])
            traj    = calculate_geodesic_trajectory(origin, dest, 120)
            spd     = round(random.uniform(10, 20), 1)
            ships   = ["MAERSK ESSEX", "MSC GULSUN", "EVER ACE",
                       "COSCO SHIPPING", "HMM ALGECIRAS",
                       "YANG MING WISH", "ONE INNOVATION", "CMA CGM MARCO POLO"]
            node = self._make_node(
                ships[i % len(ships)], "maritime", traj, spd, i * 15,
                {
                    "vessel_type":   random.choice(["Container", "Bulk Carrier", "Tanker"]),
                    "deadweight_t":  random.randint(80000, 400000),
                    "flag":          random.choice(["PAN", "LBR", "MHL", "BHS", "SGP"]),
                    "chokepoint":    cp["name"],
                    "signal_strength": round(random.uniform(0.85, 0.98), 3),
                    "imo":           f"IMO{random.randint(9100000, 9999999)}",
                }
            )
            fleet.append(node)
        return fleet

    # ── Threat ────────────────────────────────────────────────────────────

    def _build_threat_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        zones = STRATEGIC_SECTORS["conflict_zones"]
        designations = [
            "TOMAHAWK BATTERY ALPHA", "PATRIOT PAC-3 UNIT",
            "SU-35 SQUADRON BRAVO",  "F-35 STRIKE PACKAGE",
            "S-400 RADAR LOCK",      "CARRIER STRIKE GROUP 11",
            "ICBM TRANSPORTER",      "CRUISE VECTOR DELTA",
        ]
        for i, zone in enumerate(zones):
            # Patrol circuit around conflict zone center
            lat0, lng0 = zone["lat"], zone["lng"]
            r_deg = zone["radius_km"] / 111.0
            origin = (lat0 + r_deg * 0.5, lng0 - r_deg * 0.5)
            dest   = (lat0 - r_deg * 0.5, lng0 + r_deg * 0.5)
            traj   = calculate_geodesic_trajectory(origin, dest, 120)
            spd    = round(random.uniform(300, 900), 1)
            node   = self._make_node(
                designations[i % len(designations)],
                "threat", traj, spd, i * 20,
                {
                    "conflict_zone":   zone["name"],
                    "threat_intensity": round(zone["intensity"], 2),
                    "platform_type":   random.choice(["AIRCRAFT", "MISSILE", "NAVAL", "GROUND"]),
                    "engagement_range_km": random.randint(150, 1500),
                    "signal_strength": round(random.uniform(0.60, 0.95), 3),
                    "status":          "ACTIVE" if zone["intensity"] > 0.8 else "MONITORING",
                }
            )
            fleet.append(node)
        return fleet

    # ── Weather ───────────────────────────────────────────────────────────

    def _build_weather_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        for sys in STRATEGIC_SECTORS["weather_systems"]:
            lat0, lng0 = sys["lat"], sys["lng"]
            drift_lat  = lat0 + random.uniform(-2, 2)
            drift_lng  = lng0 + random.uniform(-2, 2)
            origin = (lat0,      lng0)
            dest   = (drift_lat, drift_lng + 8)
            traj   = calculate_geodesic_trajectory(origin, dest, 120)
            spd    = round(random.uniform(8, 28), 1)
            node   = self._make_node(
                sys["name"], "weather", traj, spd, 0,
                {
                    "category":           sys["category"],
                    "radius_km":          sys["radius_km"],
                    "wind_speed_knots":   random.randint(65, 140),
                    "pressure_mb":        random.randint(880, 940),
                    "storm_surge_m":      round(random.uniform(1.5, 6.0), 1),
                    "signal_strength":    1.0,
                    "status":             "ACTIVE",
                }
            )
            fleet.append(node)
        return fleet

    # ── Cyber ─────────────────────────────────────────────────────────────

    def _build_cyber_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        integrity_drop = self._threat == "OMEGA_CRITICAL"
        for cable in STRATEGIC_SECTORS["cyber_infrastructure"]:
            traj  = calculate_geodesic_trajectory(cable["origin"], cable["dest"], 120)
            spd   = round(random.uniform(45000, 75000), 0)
            integ = round(random.uniform(0.3, 0.6) if integrity_drop
                          else random.uniform(0.85, 0.99), 3)
            node  = self._make_node(
                cable["name"], "cyber", traj, spd, 0,
                {
                    "cable_type":      "Fiber Optic Undersea",
                    "bandwidth_tbps":  random.randint(30, 240),
                    "integrity":       integ,
                    "latency_ms":      random.randint(60, 180),
                    "signal_strength": integ,
                    "status":          "DEGRADED" if integ < 0.7 else "NOMINAL",
                }
            )
            fleet.append(node)
        return fleet

    # ── SIGINT ────────────────────────────────────────────────────────────

    def _build_sigint_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        for stn in STRATEGIC_SECTORS["sigint_stations"]:
            # SIGINT stations are stationary — tiny jitter path
            lat, lng = stn["lat"], stn["lng"]
            traj = [{"lat": lat + random.uniform(-0.001, 0.001),
                     "lng": lng + random.uniform(-0.001, 0.001)}
                    for _ in range(120)]
            node = self._make_node(
                stn["name"], "sigint", traj, 0.0, 0,
                {
                    "frequency_mhz":  stn["freq_mhz"],
                    "range_km":       stn["range_km"],
                    "collection_mode": random.choice(["COMINT", "ELINT", "MASINT"]),
                    "intercepts_hr":  random.randint(800, 4500),
                    "signal_strength": round(random.uniform(0.92, 0.99), 3),
                    "status":         "ACTIVE COLLECTION",
                }
            )
            fleet.append(node)
        return fleet

    # ── Commodity ─────────────────────────────────────────────────────────

    def _build_commodity_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        for flow in STRATEGIC_SECTORS["commodity_flows"]:
            traj = calculate_geodesic_trajectory(flow["origin"], flow["dest"], 120)
            spd  = round(random.uniform(12, 18), 1)
            node = self._make_node(
                flow["name"], "commodity", traj, spd, 0,
                {
                    "commodity":     flow["commodity"],
                    "volume_kbd":    flow["volume_kbd"],
                    "price_usd":     round(random.uniform(70, 110), 2),
                    "vessel_count":  random.randint(2, 12),
                    "signal_strength": round(random.uniform(0.88, 0.99), 3),
                    "market_impact": random.choice(["LOW", "MODERATE", "HIGH"]),
                }
            )
            fleet.append(node)
        return fleet

    # ── Trade ─────────────────────────────────────────────────────────────

    def _build_trade_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        for corridor in STRATEGIC_SECTORS["trade_corridors"]:
            traj = calculate_geodesic_trajectory(
                corridor["origin"], corridor["dest"], 120)
            spd  = round(random.uniform(14, 22), 1)
            node = self._make_node(
                corridor["name"], "trade", traj, spd, 0,
                {
                    "cargo_type":    random.choice(["General", "Bulk", "Container", "RoRo"]),
                    "teu_capacity":  random.randint(5000, 24000),
                    "utilisation":   round(random.uniform(0.72, 0.96), 2),
                    "signal_strength": round(random.uniform(0.88, 0.98), 3),
                    "corridor_rank": random.randint(1, 10),
                }
            )
            fleet.append(node)
        return fleet

    # ── Geo ───────────────────────────────────────────────────────────────

    def _build_geo_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        fleet = []
        for cp in STRATEGIC_SECTORS["maritime_chokepoints"][:6]:
            traj = [{"lat": cp["lat"], "lng": cp["lng"]}] * 120
            node = self._make_node(
                cp["name"], "geo", traj, 0.0, 0,
                {
                    "type":            "Strategic Chokepoint",
                    "traffic_density": cp["traffic_density"],
                    "annual_tonnage_mt": random.randint(200, 1200),
                    "signal_strength": 1.0,
                    "status":          "MONITORED",
                }
            )
            fleet.append(node)
        return fleet

    # ── Generic fallback ──────────────────────────────────────────────────

    def _build_generic_fleet(self, bbox: List[float]) -> List[Dict[str, Any]]:
        lat_c = (bbox[0] + bbox[2]) / 2
        lng_c = (bbox[1] + bbox[3]) / 2
        traj  = calculate_geodesic_trajectory(
            (lat_c - 5, lng_c - 10), (lat_c + 5, lng_c + 10), 120)
        return [self._make_node(
            "GENERIC NODE ALPHA", "geo", traj, 12.0, 0,
            {"signal_strength": 0.85, "status": "NOMINAL"}
        )]

    # ── Node factory ──────────────────────────────────────────────────────

    def _make_node(self, designation: str, domain: str,
                   trajectory: List[Dict[str, float]],
                   velocity_knots: float, start_index: int,
                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        idx = start_index % max(len(trajectory), 1)
        pos = trajectory[idx] if trajectory else {"lat": 0.0, "lng": 0.0}
        nxt = trajectory[(idx + 1) % len(trajectory)] if len(trajectory) > 1 else pos
        return {
            "node_id":          str(uuid.uuid4())[:8],
            "designation":      designation,
            "domain":           domain,
            "velocity_knots":   velocity_knots,
            "heading":          _geodesic_heading(pos, nxt),
            "current_position": pos,
            "trajectory_path":  trajectory,
            "tactical_metadata": metadata,
            "_path_index":      idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL LOG GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_LOG_TEMPLATES = [
    "[SYS]  GEODESIC ENGINE TICK @ {ts:.3f}s — nodes={n}",
    "[NET]  ADS-B INGEST STREAM OK — latency {lat}ms",
    "[SEC]  THREAT CORRELATION INDEX: {ti:.2f}",
    "[SAT]  LEO ORBITAL PASS COMPUTED — inclination {inc:.1f}°",
    "[CYBER] PACKET FLOW NOMINAL — throughput {tbps}Tbps",
    "[SIGINT] FREQ SWEEP {f}MHz — intercept confidence {c:.1f}%",
    "[MAR]  VESSEL AIS PING — MMSI {mmsi} @ {lat:.2f},{lng:.2f}",
    "[TRADE] CORRIDOR UTILISATION {u:.0f}% — ETA deviation +{d}min",
    "[WX]   STORM SYSTEM TRACK UPDATE — Δpressure -{dp}mb/hr",
    "[CMD]  DOMAIN SYNC COMPLETE — {d} nodes active",
    "[SYS]  INTEGRITY CHECK PASSED — subsystems: {s}/9 NOMINAL",
    "[THREAT] RADAR CONTACT — bearing {b:.0f}° range {r}km classification UNKNOWN",
    "[OPS]  BOUNDING BOX UPDATED [{lat1:.1f},{lng1:.1f}] to [{lat2:.1f},{lng2:.1f}]",
    "[COMMS] ENCRYPTED CHANNEL ESTABLISHED — key rotation {k}s",
    "[INTEL] PATTERN OF LIFE ANALYSIS COMPLETE — confidence {c:.0f}%",
]

def generate_terminal_log(node_count: int = 0, threat: str = "ALPHA_CLEAR") -> str:
    tmpl = random.choice(_LOG_TEMPLATES)
    return tmpl.format(
        ts   = time.time() % 10000,
        n    = node_count,
        lat  = random.randint(12, 180),
        lat1 = random.uniform(-90, 90),
        lat2 = random.uniform(-90, 90),
        lng1 = random.uniform(-180, 180),
        lng2 = random.uniform(-180, 180),
        lng  = random.uniform(-180, 180),
        ti   = random.uniform(0.3 if threat == "ALPHA_CLEAR" else 0.7, 1.0),
        inc  = random.uniform(28, 98),
        tbps = random.randint(80, 400),
        f    = random.randint(1200, 22000),
        c    = random.uniform(60, 99),
        mmsi = random.randint(200000000, 999999999),
        u    = random.uniform(60, 98),
        d    = random.randint(5, 120),
        dp   = random.randint(1, 8),
        b    = random.uniform(0, 360),
        r    = random.randint(50, 1200),
        s    = random.randint(7, 9),
        k    = random.randint(30, 300),
    )
