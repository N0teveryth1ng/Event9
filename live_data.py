"""
live_data.py — Universal Live Data Fetcher
==========================================
Adapters for free public APIs — no API keys required:
  • OpenSky Network   : real-time global flight positions
  • wheretheiss.at    : ISS live position
  • USGS              : real-time earthquake events
  • OpenMeteo         : weather conditions at a coordinate
  • Wikipedia         : descriptive summaries for anything

Each adapter returns a normalised List[LiveEntity] ready for globe projection.
"""
import logging
import math
import time
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("LiveData")

TIMEOUT = 8.0   # seconds per HTTP request


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class DataSource(str, Enum):
    OPENSKY_FLIGHTS   = "opensky_flights"
    ISS_TRACKING      = "iss_tracking"
    USGS_EARTHQUAKES  = "usgs_earthquakes"
    OPENMETEO_WEATHER = "openmeteo_weather"
    WIKIPEDIA_FACT    = "wikipedia_fact"
    MARITIME_AIS      = "maritime_ais"       # simulated AIS vessels on real lanes
    SATELLITE_TRACK   = "satellite_track"    # LEO orbital propagation
    CONFLICT_EVENTS   = "conflict_events"    # GDELT real-time conflict events
    CYBER_THREATS     = "cyber_threats"      # simulated global cyber threat nodes
    SIGINT_NODES      = "sigint_nodes"       # simulated SIGINT collection points
    GENERIC_ROUTE     = "generic_route"


class BoundingBox(BaseModel):
    lat_min: float = -90.0
    lat_max: float =  90.0
    lon_min: float = -180.0
    lon_max: float =  180.0


class DataFetchPlan(BaseModel):
    source:          DataSource
    display_label:   str         = "Live Data"
    model_key:       str         = "generic_aircraft"
    entity_color:    List[int]   = Field(default_factory=lambda: [0, 220, 255, 255])
    bbox:            Optional[BoundingBox] = None
    filter_keyword:  Optional[str]         = None   # callsign/airline/name substring
    location_query:  Optional[str]         = None   # for weather/wiki
    max_entities:    int   = 400
    refresh_s:       float = 15.0


class LiveEntity(BaseModel):
    id:          str
    label:       str
    lat:         float
    lon:         float
    alt_m:       float = 0.0
    heading_deg: float = 0.0
    speed_kmh:   float = 0.0
    entity_type: str   = "flight"    # flight | ship | earthquake | satellite | weather | iss
    model_key:   str   = "generic_aircraft"
    color:       List[int] = Field(default_factory=lambda: [0, 220, 255, 255])
    meta:        Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Universal Fetcher
# ─────────────────────────────────────────────────────────────────

class UniversalFetcher:

    async def fetch(self, plan: DataFetchPlan) -> List[LiveEntity]:
        try:
            adapters = {
                DataSource.OPENSKY_FLIGHTS:   self._opensky,
                DataSource.ISS_TRACKING:      self._iss,
                DataSource.USGS_EARTHQUAKES:  self._earthquakes,
                DataSource.OPENMETEO_WEATHER: self._weather,
                DataSource.WIKIPEDIA_FACT:    self._wikipedia,
                DataSource.MARITIME_AIS:      self._maritime,
                DataSource.SATELLITE_TRACK:   self._satellite,
                DataSource.CONFLICT_EVENTS:   self._conflict,
                DataSource.CYBER_THREATS:     self._cyber,
                DataSource.SIGINT_NODES:      self._sigint,
                DataSource.GENERIC_ROUTE:     self._generic_route,
            }
            fn = adapters.get(plan.source, self._generic_route)
            entities = await fn(plan)
            logger.info(f"[{plan.source}] fetched {len(entities)} entities")
            return entities
        except Exception as e:
            logger.error(f"Fetch failed for {plan.source}: {e}")
            return []

    # ── Flight data: adsb.lol primary → OpenSky fallback ─────────

    def _parse_readsb(self, aircraft: list, plan: DataFetchPlan, bb: BoundingBox) -> List[LiveEntity]:
        """Parse readsb-format JSON (used by adsb.lol, adsb.fi, and similar)."""
        entities: List[LiveEntity] = []
        kw = (plan.filter_keyword or "").lower()
        for ac in aircraft:
            lat = ac.get("lat")
            lon = ac.get("lon")
            if lat is None or lon is None:
                continue
            # Secondary bbox clip — radius queries can overshoot
            if not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                continue
            try:
                alt_m = float(ac.get("alt_baro") or ac.get("alt_geom") or 0) * 0.3048
            except (TypeError, ValueError):
                alt_m = 0.0
            callsign = (ac.get("flight") or "").strip() or ac.get("hex", "")
            if kw and kw not in callsign.lower():
                continue
            entities.append(LiveEntity(
                id          = ac.get("hex", f"x{len(entities)}"),
                label       = callsign,
                lat         = float(lat),
                lon         = float(lon),
                alt_m       = alt_m,
                heading_deg = float(ac.get("track") or 0),
                speed_kmh   = round(float(ac.get("gs") or 0) * 1.852, 1),
                model_key   = plan.model_key,
                color       = plan.entity_color,
                meta        = {
                    "callsign":    callsign,
                    "type":        ac.get("t", ""),
                    "reg":         ac.get("r", ""),
                    "squawk":      ac.get("squawk", ""),
                    "source":      "adsb.lol",
                },
            ))
            if len(entities) >= plan.max_entities:
                break
        return entities

    async def _opensky(self, plan: DataFetchPlan) -> List[LiveEntity]:
        bb = plan.bbox or BoundingBox()
        lat_c = (bb.lat_min + bb.lat_max) / 2
        lon_c = (bb.lon_min + bb.lon_max) / 2
        dlat_nm = (bb.lat_max - bb.lat_min) * 60.0
        dlon_nm = (bb.lon_max - bb.lon_min) * 60.0 * math.cos(math.radians(lat_c))
        radius_nm = int(math.sqrt((dlat_nm / 2) ** 2 + (dlon_nm / 2) ** 2) * 1.1) + 40

        # ── Primary: adsb.lol community ADS-B (much better global coverage) ──
        try:
            cap = min(radius_nm, 1200)
            url = (f"https://api.adsb.lol/v2/lat/{lat_c:.3f}"
                   f"/lon/{lon_c:.3f}/dist/{cap}")
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
                headers={"User-Agent": "EventNine/3.0 situational-awareness"}
            ) as client:
                r = await client.get(url)
                r.raise_for_status()
                entities = self._parse_readsb(r.json().get("ac") or [], plan, bb)
            if entities:
                logger.info(f"adsb.lol: {len(entities)} flights  radius={cap}nm")
                return entities
        except Exception as e:
            logger.warning(f"adsb.lol failed ({e}) — trying OpenSky")

        # ── Fallback: OpenSky Network bbox ────────────────────────────────────
        try:
            url = (
                "https://opensky-network.org/api/states/all"
                f"?lamin={bb.lat_min}&lomin={bb.lon_min}"
                f"&lamax={bb.lat_max}&lomax={bb.lon_max}"
            )
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
            states = data.get("states") or []
            entities = []
            kw = (plan.filter_keyword or "").lower()
            for s in states:
                if len(s) < 11: continue
                lon2, lat2, alt = s[5], s[6], s[7]
                if lon2 is None or lat2 is None or s[8]: continue
                callsign = (s[1] or s[0] or "").strip() or s[0]
                if kw and kw not in callsign.lower() and kw not in (s[2] or "").lower(): continue
                entities.append(LiveEntity(
                    id=s[0], label=callsign,
                    lat=float(lat2), lon=float(lon2),
                    alt_m=float(alt or 0),
                    heading_deg=float(s[10] or 0),
                    speed_kmh=round(float(s[9] or 0) * 3.6, 1),
                    model_key=plan.model_key, color=plan.entity_color,
                    meta={"country": s[2] or "", "source": "OpenSky"},
                ))
                if len(entities) >= plan.max_entities: break
            return entities
        except Exception as e:
            logger.error(f"OpenSky fallback failed: {e}")
            return []

    # ── ISS live position ─────────────────────────────────────────

    async def _iss(self, plan: DataFetchPlan) -> List[LiveEntity]:
        # Try wheretheiss.at first, fall back to open-notify.org
        urls = [
            ("https://api.wheretheiss.at/v1/satellites/25544", "wheretheiss"),
            ("http://api.open-notify.org/iss-now.json", "open-notify"),
        ]
        d = None
        source_name = ""
        for url, src in urls:
            try:
                async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    d = r.json()
                    source_name = src
                    break
            except Exception:
                continue

        if d is None:
            return []

        if source_name == "open-notify":
            pos = d.get("iss_position", {})
            lat = float(pos.get("latitude", 0))
            lon = float(pos.get("longitude", 0))
            alt_m, speed_kmh = 408_000.0, 27_600.0  # typical ISS values
        else:
            lat       = float(d["latitude"])
            lon       = float(d["longitude"])
            alt_m     = float(d["altitude"]) * 1000
            speed_kmh = round(float(d["velocity"]), 1)

        return [LiveEntity(
            id          = "ISS-1",
            label       = "ISS",
            lat         = lat,
            lon         = lon,
            alt_m       = alt_m,
            speed_kmh   = speed_kmh,
            heading_deg = 0.0,
            entity_type = "iss",
            model_key   = "rocket",
            color       = [255, 230, 0, 255],
            meta        = {
                "altitude_km": round(alt_m / 1000, 1),
                "velocity_kmh": round(speed_kmh, 1),
                "source": source_name,
            },
        )]

    # ── USGS real-time earthquakes ────────────────────────────────

    async def _earthquakes(self, plan: DataFetchPlan) -> List[LiveEntity]:
        # Use "all_day" for last 24h; "significant_week" for major ones
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()

        entities: List[LiveEntity] = []
        bb = plan.bbox

        for feat in data.get("features", []):
            coords = feat["geometry"]["coordinates"]  # [lon, lat, depth_km]
            props  = feat["properties"]
            lon, lat, depth = float(coords[0]), float(coords[1]), float(coords[2])

            if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                continue

            mag   = props.get("mag") or 0
            place = props.get("place") or "Unknown location"
            eid   = feat.get("id", f"eq_{len(entities)}")

            # Color-code by magnitude
            if mag >= 6.0:
                color = [255, 30,  30,  255]   # red — major
            elif mag >= 4.5:
                color = [255, 140,  0,  255]   # orange — moderate
            else:
                color = [255, 220, 50,  255]   # yellow — minor

            entities.append(LiveEntity(
                id          = eid,
                label       = f"M{mag:.1f}",
                lat         = lat,
                lon         = lon,
                alt_m       = -depth * 1000,
                entity_type = "earthquake",
                model_key   = "sphere",
                color       = color,
                meta        = {"magnitude": mag, "place": place,
                               "depth_km": depth, "source": "USGS"},
            ))
            if len(entities) >= plan.max_entities:
                break

        return entities

    # ── OpenMeteo weather — global multi-city + single location ──────

    _WMO_LABEL: Dict[int, str] = {
        0:"Clear Sky", 1:"Mainly Clear", 2:"Partly Cloudy", 3:"Overcast",
        45:"Fog", 48:"Freezing Fog",
        51:"Light Drizzle", 53:"Drizzle", 55:"Heavy Drizzle",
        56:"Freezing Drizzle", 57:"Heavy Freezing Drizzle",
        61:"Light Rain", 63:"Rain", 65:"Heavy Rain",
        66:"Freezing Rain", 67:"Heavy Freezing Rain",
        71:"Light Snow", 73:"Snow", 75:"Heavy Snow", 77:"Snow Grains",
        80:"Rain Showers", 81:"Showers", 82:"Violent Showers",
        85:"Snow Showers", 86:"Heavy Snow Showers",
        95:"Thunderstorm", 96:"Thunderstorm + Hail", 99:"Severe Thunderstorm",
    }
    _WMO_ICON: Dict[int, str] = {
        0:"☀️", 1:"🌤️", 2:"⛅", 3:"☁️",
        45:"🌫️", 48:"🌫️",
        51:"🌦️", 53:"🌧️", 55:"🌧️", 56:"🌧️", 57:"🌧️",
        61:"🌧️", 63:"🌧️", 65:"🌧️", 66:"🌧️", 67:"🌧️",
        71:"🌨️", 73:"❄️", 75:"❄️", 77:"🌨️",
        80:"🌦️", 81:"🌧️", 82:"⛈️",
        85:"🌨️", 86:"❄️",
        95:"⛈️", 96:"⛈️", 99:"🌩️",
    }

    _WEATHER_CITIES = [
        ("New York",      40.71, -74.01), ("London",       51.51,  -0.13),
        ("Tokyo",         35.69, 139.69), ("Paris",        48.85,   2.35),
        ("Sydney",       -33.87, 151.21), ("Mumbai",       19.08,  72.88),
        ("Dubai",         25.20,  55.27), ("Singapore",     1.35, 103.82),
        ("São Paulo",   -23.55, -46.63), ("Moscow",        55.75,  37.62),
        ("Lagos",          6.52,   3.38), ("Cairo",         30.05,  31.23),
        ("Mexico City",   19.43, -99.13), ("Beijing",       39.90, 116.41),
        ("Jakarta",       -6.21, 106.85), ("Berlin",        52.52,  13.40),
        ("Buenos Aires", -34.60, -58.38), ("Karachi",       24.86,  67.01),
        ("Istanbul",      41.01,  28.95), ("Seoul",         37.57, 126.98),
        ("Manila",        14.60, 120.98), ("Kinshasa",      -4.32,  15.33),
        ("Lima",         -12.04, -77.03), ("Baghdad",       33.34,  44.40),
        ("Riyadh",        24.69,  46.72), ("Los Angeles",   34.05,-118.24),
        ("Chicago",       41.85, -87.65), ("Bangkok",       13.75, 100.52),
        ("Toronto",       43.65, -79.38), ("Johannesburg", -26.20,  28.04),
        ("Taipei",        25.05, 121.53), ("Nairobi",       -1.29,  36.82),
        ("Frankfurt",     50.11,   8.68), ("Miami",         25.77, -80.19),
        ("Osaka",         34.69, 135.50), ("Bogotá",         4.71, -74.07),
        ("Houston",       29.76, -95.37), ("Guangzhou",     23.13, 113.26),
        ("Dhaka",         23.72,  90.41), ("Lahore",        31.55,  74.34),
    ]

    async def _weather(self, plan: DataFetchPlan) -> List[LiveEntity]:
        bb = plan.bbox
        if plan.location_query:
            lat, lon = await self._geocode(plan.location_query)
            cities = [(plan.location_query.title(), lat, lon)]
        elif bb and (bb.lat_max - bb.lat_min) < 15:
            lat = (bb.lat_min + bb.lat_max) / 2
            lon = (bb.lon_min + bb.lon_max) / 2
            cities = [(f"{lat:.1f}°, {lon:.1f}°", lat, lon)]
        else:
            cities = [
                (name, la, lo) for name, la, lo in self._WEATHER_CITIES
                if not bb or (bb.lat_min <= la <= bb.lat_max and bb.lon_min <= lo <= bb.lon_max)
            ] or list(self._WEATHER_CITIES)

        lats = ",".join(f"{la:.2f}" for _, la, _ in cities)
        lons = ",".join(f"{lo:.2f}" for _, _, lo in cities)
        # Use the richer `current` API for full weather metrics
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lats}&longitude={lons}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
            "wind_gusts_10m,visibility,uv_index,cloud_cover,surface_pressure"
            "&wind_speed_unit=kmh&timezone=auto"
        )
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            logger.error(f"OpenMeteo failed: {e}")
            return []

        if isinstance(data, dict):
            data = [data]

        entities: List[LiveEntity] = []
        for i, d in enumerate(data):
            if i >= len(cities):
                break
            name, lat, lon = cities[i]
            cur  = d.get("current", {})

            temp     = float(cur.get("temperature_2m") or 0)
            feels    = float(cur.get("apparent_temperature") or temp)
            humidity = int(cur.get("relative_humidity_2m") or 0)
            wind     = float(cur.get("wind_speed_10m") or 0)
            gusts    = float(cur.get("wind_gusts_10m") or 0)
            wind_dir = int(cur.get("wind_direction_10m") or 0)
            precip   = float(cur.get("precipitation") or 0)
            wcode    = int(cur.get("weather_code") or 0)
            uv       = float(cur.get("uv_index") or 0)
            cloud    = int(cur.get("cloud_cover") or 0)
            vis_m    = float(cur.get("visibility") or 10000)
            pressure = float(cur.get("surface_pressure") or 1013)

            condition = self._WMO_LABEL.get(wcode, f"Code {wcode}")
            icon_e    = self._WMO_ICON.get(wcode, "🌡️")

            # Color by severity
            if wcode >= 95 or wind > 80:
                color = [255, 61, 61, 230]
            elif wcode >= 80 or wind > 50 or precip > 5:
                color = [255, 140, 0, 220]
            elif wcode >= 50:
                color = [100, 180, 255, 210]
            elif wcode >= 3:
                color = [180, 180, 180, 200]
            else:
                color = [255, 214, 0, 200]

            # Wind direction cardinal
            dirs   = ["N","NE","E","SE","S","SW","W","NW"]
            card   = dirs[round(wind_dir / 45) % 8]

            entities.append(LiveEntity(
                id          = f"wx_{lat:.2f}_{lon:.2f}",
                label       = f"{icon_e} {name}  {temp:+.0f}°C",
                lat         = lat,
                lon         = lon,
                alt_m       = 0.0,
                heading_deg = float(wind_dir),
                speed_kmh   = wind,
                entity_type = "weather",
                model_key   = "sphere",
                color       = color,
                meta        = {
                    "condition":      condition,
                    "temperature":    f"{temp:.1f}°C",
                    "feels like":     f"{feels:.1f}°C",
                    "humidity":       f"{humidity}%",
                    "wind":           f"{wind:.0f} km/h {card}",
                    "gusts":          f"{gusts:.0f} km/h",
                    "precipitation":  f"{precip:.1f} mm",
                    "cloud cover":    f"{cloud}%",
                    "visibility":     f"{vis_m/1000:.1f} km",
                    "UV index":       f"{uv:.1f}",
                    "pressure":       f"{pressure:.0f} hPa",
                    "source":         "Open-Meteo",
                },
            ))
        logger.info(f"OpenMeteo: {len(entities)} weather stations")
        return entities

    # ── Wikipedia summary fetch ───────────────────────────────────

    async def _wikipedia(self, plan: DataFetchPlan) -> List[LiveEntity]:
        """Returns descriptive metadata — used alongside 3D model mode."""
        query = plan.location_query or plan.display_label
        url   = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    d = r.json()
                    return [LiveEntity(
                        id    = "wiki_1",
                        label = d.get("title", query),
                        lat   = 0.0, lon = 0.0,
                        meta  = {
                            "summary":   d.get("extract", "")[:400],
                            "thumbnail": d.get("thumbnail", {}).get("source", ""),
                            "source":    "Wikipedia",
                        },
                    )]
        except Exception:
            pass
        return []

    # ── Maritime AIS simulation ───────────────────────────────────

    # Major shipping lanes: (from_lat, from_lon, to_lat, to_lon, count, vtype, speed_kts)
    _LANES = [
        (1.3, 103.8, 22.3, 114.2, 18, "Container", 18),   # Singapore → HK
        (26.0, 50.5, 1.3,  103.8, 22, "Tanker",    14),   # Persian Gulf → Singapore
        (31.2, 121.5, 37.9,  126.9, 12, "Bulk",    12),   # Shanghai → Busan
        (51.5, -0.1,  40.7,  -74.0, 16, "Container",17),  # London → New York
        (29.9, 32.6,  12.8,  45.0,  10, "Tanker",   15),  # Suez → Gulf of Aden
        (-33.9, 18.4, 22.3, 114.2,  8, "Bulk",     13),   # Cape Town → HK
        (1.3, 103.8, -33.9,  151.2, 14, "Container",16),  # Singapore → Sydney
        (40.7, -74.0, 48.9,   2.3,  10, "RoRo",    19),   # NY → Le Havre
        (36.0, -5.6,  51.5,  -0.1,  12, "Tanker",   14),  # Gibraltar → London
        (22.3, 114.2, 35.7,  139.7, 16, "Container",17),  # HK → Tokyo
        (60.0, 5.3,   51.5,  -0.1,  8,  "Tanker",   13),  # Norway → UK
        (-23.5,-43.2, 51.5,  -0.1, 10,  "Bulk",     12),  # Rio → London
        (26.0, 50.5,  51.5,  -0.1, 12,  "Tanker",   14),  # Persian Gulf → UK
        (13.0, 80.3,  1.3,  103.8, 10,  "Container",16),  # Chennai → Singapore
    ]

    _VESSEL_TYPES = {
        "Container": ([0, 180, 255, 220], "container ship"),
        "Tanker":    ([255, 140, 0,  220], "crude oil tanker"),
        "Bulk":      ([100, 200, 100, 220], "bulk carrier"),
        "RoRo":      ([200, 150, 255, 220], "ro-ro vessel"),
    }

    async def _maritime(self, plan: DataFetchPlan) -> List[LiveEntity]:
        """
        Simulates vessels on major global shipping lanes using time-based
        interpolation — positions are deterministic, advancing realistically.
        """
        t_now = time.time()
        bb    = plan.bbox
        entities: List[LiveEntity] = []

        for lane_idx, (lat1, lon1, lat2, lon2, n, vtype, speed_kts) in enumerate(self._LANES):
            # Great-circle distance (km)
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            dist_km = 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

            # Transit time in seconds
            transit_s = (dist_km / (speed_kts * 1.852)) * 3600

            color, vdesc = self._VESSEL_TYPES.get(vtype, ([200,200,200,220], "vessel"))

            for i in range(n):
                # Each vessel offset in phase so they're spread along the lane
                phase_offset = (i / n) * transit_s
                t_frac = ((t_now + phase_offset) % transit_s) / transit_s

                # Interpolate position along great circle (linear approx for short segments)
                lat = lat1 + (lat2 - lat1) * t_frac
                lon = lon1 + (lon2 - lon1) * t_frac

                if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                    continue

                # Heading from start to end
                dlon_r = math.radians(lon2 - lon1)
                heading = math.degrees(math.atan2(
                    math.sin(dlon_r) * math.cos(math.radians(lat2)),
                    math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
                    math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon_r)
                )) % 360

                # Realistic variation: small noise on speed/heading
                speed_noise  = math.sin(t_now * 0.001 + i * 7.3) * 0.8
                heading_noise = math.sin(t_now * 0.0007 + i * 3.1) * 2.5

                mmsi = 300_000_000 + lane_idx * 100 + i
                entities.append(LiveEntity(
                    id          = f"ship_{mmsi}",
                    label       = f"{vtype[:3].upper()}-{mmsi % 10000:04d}",
                    lat         = lat,
                    lon         = lon,
                    alt_m       = 0.0,
                    heading_deg = (heading + heading_noise) % 360,
                    speed_kmh   = round((speed_kts + speed_noise) * 1.852, 1),
                    model_key   = "ship",
                    color       = color,
                    entity_type = "ship",
                    meta        = {
                        "mmsi":    mmsi,
                        "type":    vdesc,
                        "flag":    "INT",
                        "draft_m": round(8 + (i % 5) * 1.5, 1),
                        "source":  "AIS-SIM",
                    },
                ))
                if len(entities) >= plan.max_entities:
                    return entities

        logger.info(f"Maritime AIS simulation: {len(entities)} vessels")
        return entities

    # ── Satellite orbital tracking ────────────────────────────────

    # LEO / MEO satellites: (name, alt_km, inc_deg, period_min, raan_phase, color)
    _SATS = [
        # GPS constellation (MEO)
        *[(f"GPS-{i+1}", 20200, 55, 718, i*(360/24), [0,255,180,220]) for i in range(24)],
        # Starlink (LEO) — 6 planes
        *[(f"STARLINK-{i+1}", 550, 53, 95.5, i*(360/60), [100,180,255,200]) for i in range(60)],
        # Galileo (MEO)
        *[(f"GALILEO-{i+1}", 23222, 56, 844, i*(360/30), [255,200,0,220]) for i in range(30)],
        # ISS is handled separately
        ("NOAA-15",  807, 98.7, 101.2, 45,  [0,255,100,220]),
        ("NOAA-18",  854, 99.0, 102.1, 90,  [0,255,100,220]),
        ("NOAA-19",  870, 99.0, 102.1, 135, [0,255,100,220]),
        ("TERRA",    705, 98.2, 98.9,  180, [255,160,0,220]),
        ("AQUA",     705, 98.2, 98.9,  225, [0,200,255,220]),
        ("SENTINEL-1",693, 98.2, 98.6, 270, [255,80,80,220]),
        ("SENTINEL-2",786, 98.6, 100.6,315, [80,255,80,220]),
    ]

    @staticmethod
    def _orbit_latlon(alt_km: float, inc_deg: float, period_min: float,
                       raan_phase: float) -> Tuple[float, float]:
        """
        Simplified sub-satellite point (lat, lon) for a circular orbit.
        Not SGP4 — accurate enough for visual display.
        """
        t   = time.time()
        inc = math.radians(inc_deg)
        # Mean motion (rad/s)
        n   = 2 * math.pi / (period_min * 60)
        # Current argument of latitude
        theta = (n * t + math.radians(raan_phase * 3.7)) % (2 * math.pi)
        # Sub-satellite latitude
        lat = math.degrees(math.asin(math.sin(inc) * math.sin(theta)))
        # Longitude: include Earth rotation (7.2921e-5 rad/s)
        lon_orbit = math.degrees(math.atan2(
            math.cos(inc) * math.sin(theta), math.cos(theta)
        ))
        earth_rot = math.degrees(7.2921e-5 * t) % 360
        raan_deg  = raan_phase % 360
        lon = ((lon_orbit + raan_deg - earth_rot) % 360 + 360) % 360
        if lon > 180:
            lon -= 360
        return lat, lon

    async def _satellite(self, plan: DataFetchPlan) -> List[LiveEntity]:
        bb = plan.bbox
        kw = (plan.filter_keyword or "").lower()
        entities: List[LiveEntity] = []

        for name, alt_km, inc, period, raan, color in self._SATS:
            if kw and kw not in name.lower():
                continue
            try:
                lat, lon = self._orbit_latlon(alt_km, inc, period, raan)
            except Exception:
                continue

            if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                continue

            entities.append(LiveEntity(
                id          = f"sat_{name.lower().replace('-','_')}",
                label       = name,
                lat         = lat,
                lon         = lon,
                alt_m       = alt_km * 1000,
                heading_deg = 0.0,
                speed_kmh   = round(math.sqrt(398600 / (6371 + alt_km)) * 3600, 0),  # km/s → km/h
                model_key   = "satellite",
                color       = color,
                entity_type = "satellite",
                meta        = {
                    "altitude_km": alt_km,
                    "inclination": inc,
                    "period_min":  period,
                    "source":      "orbital-propagation",
                },
            ))
            if len(entities) >= plan.max_entities:
                break

        logger.info(f"Satellite tracking: {len(entities)} objects")
        return entities

    # ── GDELT conflict events (real-time, no key) ─────────────────

    async def _conflict(self, plan: DataFetchPlan) -> List[LiveEntity]:
        """
        GDELT 2.0 Geo API — returns geo-tagged news events filtered to
        conflict/military keywords. Free, no API key required.
        """
        bb  = plan.bbox
        kw  = plan.filter_keyword or "conflict military airstrike attack"
        url = (
            "https://api.gdeltproject.org/api/v2/geo/geo"
            f"?query={kw.replace(' ', '+')}&TIMESPAN=1DAY&MAXROWS=80&format=geojson"
        )
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT,
                headers={"User-Agent": "EventNine/3.0 research-dashboard"}) as client:
                r = await client.get(url)
                r.raise_for_status()
                gj = r.json()
        except Exception as e:
            logger.error(f"GDELT conflict fetch failed: {e}")
            return self._conflict_fallback(bb)

        entities: List[LiveEntity] = []
        for feat in gj.get("features", []):
            try:
                lon, lat = feat["geometry"]["coordinates"]
                props = feat.get("properties", {})
                name  = props.get("name", "Conflict Event")[:40]
                url_  = props.get("url", "")
                if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                    continue
                eid = f"conflict_{lat:.3f}_{lon:.3f}"
                entities.append(LiveEntity(
                    id          = eid,
                    label       = name,
                    lat         = lat,
                    lon         = lon,
                    alt_m       = 0.0,
                    entity_type = "conflict",
                    model_key   = "sphere",
                    color       = [255, 61, 61, 220],
                    meta        = {"event": name, "url": url_[:80], "source": "GDELT"},
                ))
                if len(entities) >= plan.max_entities:
                    break
            except Exception:
                continue
        if not entities:
            return self._conflict_fallback(bb)
        logger.info(f"GDELT: {len(entities)} conflict events")
        return entities

    def _conflict_fallback(self, bb) -> List[LiveEntity]:
        """Known active conflict zones — used when GDELT is unavailable."""
        ZONES = [
            ("Kyiv Front",        50.45,  30.52), ("Kharkiv",         49.99,  36.23),
            ("Zaporizhzhia",      47.84,  35.14), ("Gaza",            31.51,  34.47),
            ("West Bank",         32.06,  35.30), ("Khartoum",        15.60,  32.53),
            ("Tigray",            14.05,  38.32), ("Sahel Region",    15.00,   0.00),
            ("Somalia",            2.05,  45.34), ("Yemen Sanaa",     15.35,  44.21),
            ("Mosul",             36.34,  43.14), ("Homs",            34.73,  36.71),
            ("Kabul",             34.53,  69.17), ("Myanmar Rakhine",  18.15,  94.20),
            ("Donbas Line",       48.00,  37.80), ("Avdiivka",        48.13,  37.75),
        ]
        entities = []
        for name, lat, lon in ZONES:
            if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                continue
            entities.append(LiveEntity(
                id=f"cz_{lat:.2f}_{lon:.2f}", label=name,
                lat=lat, lon=lon, entity_type="conflict",
                model_key="sphere", color=[255, 61, 61, 220],
                meta={"source": "known-zones"},
            ))
        return entities

    # ── Simulated cyber threat nodes ─────────────────────────────

    # Nation-state APT + major internet exchange hotspots
    _CYBER_NODES = [
        # APT clusters — China
        ("APT41 Guangdong",  23.13, 113.26, "CN-APT"),
        ("APT10 Shanghai",   31.22, 121.46, "CN-APT"),
        ("APT3 Beijing",     39.90, 116.41, "CN-APT"),
        # APT — Russia
        ("Sandworm Moscow",  55.75,  37.62, "RU-APT"),
        ("Fancy Bear SPb",   59.93,  30.32, "RU-APT"),
        ("Cozy Bear Kazan",  55.78,  49.12, "RU-APT"),
        # APT — North Korea
        ("Lazarus Pyongyang",39.02, 125.75, "DPRK-APT"),
        ("Kimsuky",          37.50, 127.02, "DPRK-APT"),
        # APT — Iran
        ("APT33 Tehran",     35.69,  51.42, "IR-APT"),
        ("Charming Kitten",  32.09,  34.78, "IR-APT"),
        # Ransomware clusters — Eastern Europe
        ("RaaS Hub Kyiv",    50.45,  30.52, "RaaS"),
        ("RaaS Minsk",       53.90,  27.57, "RaaS"),
        ("RaaS Odesa",       46.48,  30.73, "RaaS"),
        # Major IXP targets
        ("DE-CIX Frankfurt", 50.11,   8.68, "IXP"),
        ("LINX London",      51.51,  -0.13, "IXP"),
        ("AMS-IX Amsterdam", 52.37,   4.90, "IXP"),
        ("EQUINIX Ashburn",  39.01, -77.49, "IXP"),
        ("JPIX Tokyo",       35.69, 139.69, "IXP"),
        ("SG-IX Singapore",   1.35, 103.82, "IXP"),
        # Botnets
        ("Botnet-C2 Brazil", -23.55, -46.63, "BotC2"),
        ("Botnet-C2 India",   28.61,  77.23, "BotC2"),
        ("Botnet-C2 Turkey",  41.01,  28.95, "BotC2"),
    ]

    async def _cyber(self, plan: DataFetchPlan) -> List[LiveEntity]:
        bb   = plan.bbox
        t    = time.time()
        entities: List[LiveEntity] = []
        for i, (name, base_lat, base_lon, ctype) in enumerate(self._CYBER_NODES):
            # Small position jitter to simulate moving threat actors
            lat = base_lat + math.sin(t * 0.0003 + i * 1.7) * 0.15
            lon = base_lon + math.cos(t * 0.0004 + i * 2.3) * 0.15
            if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                continue
            # Pulse "activity" metric
            activity = 0.4 + 0.6 * abs(math.sin(t * 0.001 + i * 0.9))
            color = [0, 230, 118, int(180 + activity * 75)]  # green, brighter = more active
            entities.append(LiveEntity(
                id          = f"cyber_{i}",
                label       = name,
                lat         = lat,
                lon         = lon,
                entity_type = "cyber",
                model_key   = "sphere",
                color       = color,
                meta        = {
                    "type":          ctype,
                    "activity":      f"{activity*100:.0f}%",
                    "threat_level":  "HIGH" if activity > 0.7 else "MEDIUM",
                    "source":        "CyberInt-SIM",
                },
            ))
        logger.info(f"Cyber threats: {len(entities)} nodes")
        return entities

    # ── Simulated SIGINT collection nodes ────────────────────────

    # Real-world signals intelligence facilities (open-source locations)
    _SIGINT_STATIONS = [
        # NSA / Five Eyes
        ("NSA Utah DC",       40.85,-111.90, "COMINT"),
        ("NSA Ft Meade",      39.11, -76.77, "COMINT"),
        ("GCHQ Cheltenham",   51.90,  -2.09, "GCHQ"),
        ("Pine Gap AUS",     -23.80, 133.74, "ECHELON"),
        ("Bad Aibling DE",    47.87,  11.99, "NSA/BND"),
        ("Menwith Hill UK",   54.00,  -1.69, "NSA"),
        ("Misawa JP",         40.70, 141.37, "NSA-SIGINT"),
        ("RAF Ayios CYPRUS",  34.97,  33.00, "GCHQ"),
        # Russian SIGINT
        ("GRU Moscow",        55.75,  37.62, "GRU-SIGINT"),
        ("FAPSI Ekaterinburg",56.83,  60.60, "FSO"),
        ("Lourdes CUBA",      22.95, -82.21, "GRU"),
        ("Cam Ranh Bay VN",   12.00, 109.21, "GRU"),
        # Chinese SIGINT
        ("PLA SSF Beijing",   39.90, 116.41, "PLA-SSF"),
        ("PLA Hainan",        18.25, 109.50, "PLA-SSF"),
        ("PLA Xinjiang",      39.47,  75.98, "PLA-SSF"),
        # Other
        ("ISNU Tehran",       35.69,  51.42, "SIGINT"),
        ("8200 Israel",       32.07,  34.78, "Unit-8200"),
        ("DGI Paris",         48.85,   2.35, "DGSI"),
        ("BND Pullach DE",    48.06,  11.50, "BND"),
        ("SVR Yasenevo",      55.58,  37.48, "SVR"),
    ]

    async def _sigint(self, plan: DataFetchPlan) -> List[LiveEntity]:
        bb = plan.bbox
        t  = time.time()
        entities: List[LiveEntity] = []
        for i, (name, lat, lon, org) in enumerate(self._SIGINT_STATIONS):
            if bb and not (bb.lat_min <= lat <= bb.lat_max and bb.lon_min <= lon <= bb.lon_max):
                continue
            # Intercept pulse — simulates antenna sweep activity
            sweep = abs(math.sin(t * 0.0008 + i * 1.4))
            entities.append(LiveEntity(
                id          = f"sigint_{i}",
                label       = name,
                lat         = lat,
                lon         = lon,
                heading_deg = (t * 15 + i * 37) % 360,  # rotating sweep heading
                entity_type = "sigint",
                model_key   = "sphere",
                color       = [233, 30, 99, int(160 + sweep * 95)],  # magenta
                meta        = {
                    "organization":  org,
                    "sweep_coverage": f"{int(sweep*100)}%",
                    "intercepts_24h": int(1200 + sweep * 3800 + i * 290),
                    "source":        "OSINT-SIM",
                },
            ))
        logger.info(f"SIGINT nodes: {len(entities)} stations")
        return entities

    # ── Generic synthetic route (fallback) ───────────────────────

    async def _generic_route(self, plan: DataFetchPlan) -> List[LiveEntity]:
        """Returns a single synthetic waypoint at bbox center — safe fallback."""
        lat = (plan.bbox.lat_min + plan.bbox.lat_max) / 2 if plan.bbox else 0.0
        lon = (plan.bbox.lon_min + plan.bbox.lon_max) / 2 if plan.bbox else 0.0
        return [LiveEntity(
            id=f"generic_{plan.source}",
            label=plan.display_label,
            lat=lat, lon=lon,
            model_key=plan.model_key,
            color=plan.entity_color,
            meta={"note": "No live API available — synthetic position"},
        )]

    # ── Geocoder helper (Nominatim, free) ────────────────────────

    async def _geocode(self, place: str) -> Tuple[float, float]:
        url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json&limit=1"
        headers = {"User-Agent": "EventNine/2.0 situational-awareness-dashboard"}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
                r = await client.get(url)
                results = r.json()
                if results:
                    return float(results[0]["lat"]), float(results[0]["lon"])
        except Exception:
            pass
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────
# Named region bounding boxes (used by orchestrator rule-based fallback)
# ─────────────────────────────────────────────────────────────────

KNOWN_REGIONS: Dict[str, BoundingBox] = {
    # ── Continents / macro-regions ────────────────────────────────────────
    "global":           BoundingBox(lat_min=-90, lat_max=90,  lon_min=-180, lon_max=180),
    "north america":    BoundingBox(lat_min=15,  lat_max=72,  lon_min=-170, lon_max=-50),
    "south america":    BoundingBox(lat_min=-56, lat_max=13,  lon_min=-82,  lon_max=-34),
    "europe":           BoundingBox(lat_min=36,  lat_max=71,  lon_min=-10,  lon_max=42),
    "asia":             BoundingBox(lat_min=10,  lat_max=55,  lon_min=60,   lon_max=150),
    "africa":           BoundingBox(lat_min=-35, lat_max=37,  lon_min=-18,  lon_max=52),
    "middle east":      BoundingBox(lat_min=15,  lat_max=42,  lon_min=33,   lon_max=65),
    "southeast asia":   BoundingBox(lat_min=-10, lat_max=25,  lon_min=95,   lon_max=145),
    "oceania":          BoundingBox(lat_min=-50, lat_max=5,   lon_min=110,  lon_max=180),

    # ── Countries — Americas ──────────────────────────────────────────────
    "usa":              BoundingBox(lat_min=24,  lat_max=50,  lon_min=-125, lon_max=-65),
    "united states":    BoundingBox(lat_min=24,  lat_max=50,  lon_min=-125, lon_max=-65),
    "canada":           BoundingBox(lat_min=42,  lat_max=83,  lon_min=-142, lon_max=-52),
    "mexico":           BoundingBox(lat_min=14,  lat_max=32,  lon_min=-118, lon_max=-86),
    "brazil":           BoundingBox(lat_min=-33, lat_max=5,   lon_min=-74,  lon_max=-35),
    "argentina":        BoundingBox(lat_min=-55, lat_max=-22, lon_min=-73,  lon_max=-53),
    "colombia":         BoundingBox(lat_min=-4,  lat_max=13,  lon_min=-79,  lon_max=-67),
    "chile":            BoundingBox(lat_min=-56, lat_max=-17, lon_min=-76,  lon_max=-66),
    "peru":             BoundingBox(lat_min=-18, lat_max=0,   lon_min=-81,  lon_max=-68),
    "caribbean":        BoundingBox(lat_min=10,  lat_max=27,  lon_min=-85,  lon_max=-60),

    # ── Countries — Europe ────────────────────────────────────────────────
    "uk":               BoundingBox(lat_min=49,  lat_max=61,  lon_min=-9,   lon_max=2),
    "united kingdom":   BoundingBox(lat_min=49,  lat_max=61,  lon_min=-9,   lon_max=2),
    "britain":          BoundingBox(lat_min=49,  lat_max=61,  lon_min=-9,   lon_max=2),
    "germany":          BoundingBox(lat_min=47,  lat_max=55,  lon_min=6,    lon_max=15),
    "france":           BoundingBox(lat_min=42,  lat_max=51,  lon_min=-5,   lon_max=9),
    "spain":            BoundingBox(lat_min=36,  lat_max=44,  lon_min=-9,   lon_max=5),
    "italy":            BoundingBox(lat_min=36,  lat_max=47,  lon_min=7,    lon_max=18),
    "portugal":         BoundingBox(lat_min=37,  lat_max=42,  lon_min=-9,   lon_max=-6),
    "netherlands":      BoundingBox(lat_min=50,  lat_max=54,  lon_min=3,    lon_max=7),
    "belgium":          BoundingBox(lat_min=49,  lat_max=52,  lon_min=2,    lon_max=6),
    "switzerland":      BoundingBox(lat_min=46,  lat_max=48,  lon_min=6,    lon_max=10),
    "austria":          BoundingBox(lat_min=46,  lat_max=49,  lon_min=9,    lon_max=17),
    "poland":           BoundingBox(lat_min=49,  lat_max=55,  lon_min=14,   lon_max=24),
    "ukraine":          BoundingBox(lat_min=44,  lat_max=53,  lon_min=22,   lon_max=40),
    "russia":           BoundingBox(lat_min=41,  lat_max=72,  lon_min=20,   lon_max=180),
    "scandinavia":      BoundingBox(lat_min=55,  lat_max=72,  lon_min=4,    lon_max=32),
    "norway":           BoundingBox(lat_min=57,  lat_max=72,  lon_min=4,    lon_max=31),
    "sweden":           BoundingBox(lat_min=55,  lat_max=69,  lon_min=11,   lon_max=24),
    "finland":          BoundingBox(lat_min=59,  lat_max=70,  lon_min=20,   lon_max=31),
    "greece":           BoundingBox(lat_min=35,  lat_max=42,  lon_min=20,   lon_max=28),
    "turkey":           BoundingBox(lat_min=36,  lat_max=42,  lon_min=26,   lon_max=45),
    "balkans":          BoundingBox(lat_min=39,  lat_max=47,  lon_min=13,   lon_max=29),

    # ── Countries — Asia ──────────────────────────────────────────────────
    "japan":            BoundingBox(lat_min=24,  lat_max=46,  lon_min=122,  lon_max=150),
    "china":            BoundingBox(lat_min=18,  lat_max=54,  lon_min=73,   lon_max=135),
    "india":            BoundingBox(lat_min=6,   lat_max=37,  lon_min=67,   lon_max=98),
    "south korea":      BoundingBox(lat_min=33,  lat_max=39,  lon_min=124,  lon_max=130),
    "korea":            BoundingBox(lat_min=33,  lat_max=43,  lon_min=124,  lon_max=131),
    "north korea":      BoundingBox(lat_min=37,  lat_max=43,  lon_min=124,  lon_max=131),
    "indonesia":        BoundingBox(lat_min=-11, lat_max=6,   lon_min=95,   lon_max=141),
    "malaysia":         BoundingBox(lat_min=1,   lat_max=8,   lon_min=100,  lon_max=120),
    "thailand":         BoundingBox(lat_min=5,   lat_max=21,  lon_min=97,   lon_max=106),
    "vietnam":          BoundingBox(lat_min=8,   lat_max=24,  lon_min=102,  lon_max=110),
    "philippines":      BoundingBox(lat_min=5,   lat_max=21,  lon_min=117,  lon_max=127),
    "singapore":        BoundingBox(lat_min=1,   lat_max=2,   lon_min=103,  lon_max=105),
    "taiwan":           BoundingBox(lat_min=21,  lat_max=26,  lon_min=119,  lon_max=123),
    "hong kong":        BoundingBox(lat_min=22,  lat_max=23,  lon_min=113,  lon_max=115),
    "pakistan":         BoundingBox(lat_min=23,  lat_max=37,  lon_min=61,   lon_max=77),
    "bangladesh":       BoundingBox(lat_min=20,  lat_max=27,  lon_min=88,   lon_max=93),
    "sri lanka":        BoundingBox(lat_min=6,   lat_max=10,  lon_min=79,   lon_max=82),

    # ── Countries — Middle East ───────────────────────────────────────────
    "saudi arabia":     BoundingBox(lat_min=16,  lat_max=32,  lon_min=36,   lon_max=56),
    "uae":              BoundingBox(lat_min=22,  lat_max=27,  lon_min=51,   lon_max=57),
    "iran":             BoundingBox(lat_min=25,  lat_max=40,  lon_min=44,   lon_max=64),
    "iraq":             BoundingBox(lat_min=29,  lat_max=38,  lon_min=38,   lon_max=49),
    "israel":           BoundingBox(lat_min=29,  lat_max=34,  lon_min=34,   lon_max=36),
    "egypt":            BoundingBox(lat_min=22,  lat_max=32,  lon_min=25,   lon_max=37),
    "persian gulf":     BoundingBox(lat_min=22,  lat_max=30,  lon_min=48,   lon_max=60),

    # ── Countries — Africa ────────────────────────────────────────────────
    "nigeria":          BoundingBox(lat_min=4,   lat_max=14,  lon_min=3,    lon_max=15),
    "south africa":     BoundingBox(lat_min=-35, lat_max=-22, lon_min=17,   lon_max=33),
    "kenya":            BoundingBox(lat_min=-5,  lat_max=5,   lon_min=33,   lon_max=42),
    "ethiopia":         BoundingBox(lat_min=3,   lat_max=15,  lon_min=33,   lon_max=48),
    "morocco":          BoundingBox(lat_min=27,  lat_max=36,  lon_min=-14,  lon_max=-1),

    # ── Countries — Oceania ───────────────────────────────────────────────
    "australia":        BoundingBox(lat_min=-44, lat_max=-10, lon_min=113,  lon_max=154),
    "new zealand":      BoundingBox(lat_min=-47, lat_max=-34, lon_min=166,  lon_max=178),

    # ── Sea / route regions ───────────────────────────────────────────────
    "atlantic":         BoundingBox(lat_min=15,  lat_max=65,  lon_min=-65,  lon_max=-5),
    "pacific":          BoundingBox(lat_min=-20, lat_max=60,  lon_min=130,  lon_max=-120),
    "indian ocean":     BoundingBox(lat_min=-40, lat_max=25,  lon_min=40,   lon_max=100),
    "mediterranean":    BoundingBox(lat_min=30,  lat_max=47,  lon_min=-6,   lon_max=37),
    "north sea":        BoundingBox(lat_min=51,  lat_max=62,  lon_min=-5,   lon_max=10),
    "english channel":  BoundingBox(lat_min=49,  lat_max=52,  lon_min=-5,   lon_max=3),
    "suez canal":       BoundingBox(lat_min=28,  lat_max=32,  lon_min=31,   lon_max=34),
    "strait of malacca":BoundingBox(lat_min=1,   lat_max=6,   lon_min=99,   lon_max=105),
}
