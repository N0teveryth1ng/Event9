import os, asyncio, time, logging, json, uuid
import httpx
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from typing import Dict, Any, List, Optional
from google import genai as genai_sdk

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from orchestrator import route_intent, UniversalIntent, IntentMode, AssetIntent, AssetClass, parse_operational_intent
from mesh_processor import GeometryProcessor
from live_data import UniversalFetcher, DataFetchPlan, LiveEntity
from scraper import FleetSimulator, generate_terminal_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EventNine")

# ── Hardcoded geographic routes (accurate waypoints) ─────────────
KNOWN_ROUTES: Dict[str, Any] = {
    "silk road": {
        "flyto": {"lat": 38.0, "lon": 72.0, "zoom": 3},
        "routes": [
            {"label": "Northern Silk Road", "color": "#00ff66", "width": 2, "points": [
                [34.27,108.95],[36.06,103.83],[40.14,94.66],[42.82,93.51],
                [42.93,89.18],[43.83,87.62],[39.47,75.99],[40.52,72.79],
                [39.65,66.96],[39.77,64.42],[37.63,62.18],[36.21,58.79],
                [35.69,51.39],[38.08,46.29],[41.00,39.73],[39.93,32.85],
                [41.01,28.97],[41.90,12.50],
            ]},
            {"label": "Southern Silk Road", "color": "#ffb800", "width": 2, "points": [
                [34.27,108.95],[30.57,104.07],[27.71,85.31],[25.60,85.14],
                [33.74,72.84],[34.01,71.57],[34.53,69.17],[31.62,65.71],
                [34.35,62.20],[36.27,59.61],[29.59,52.58],[33.09,44.58],
                [34.55,38.27],[33.51,36.29],[36.20,36.16],[33.27,35.20],
                [31.20,29.92],
            ]},
            {"label": "Maritime Silk Road", "color": "#00bfff", "width": 2, "points": [
                [24.87,118.68],[22.27,114.17],[10.82,106.63],[2.19,102.25],
                [1.35,103.82],[6.93,79.85],[11.25,75.78],[27.07,56.46],
                [12.78,45.02],[11.59,43.14],[2.05,45.34],[-4.05,39.67],
                [30.06,31.25],[31.20,29.92],[36.89,10.33],[38.12,13.36],
                [41.90,12.50],
            ]},
        ],
        "markers": [
            {"lat":34.27,"lon":108.95,"label":"Xi'an","color":"#00ff66","info":"Eastern terminus of the Silk Road"},
            {"lat":39.47,"lon":75.99,"label":"Kashgar","color":"#00ff66","info":"Junction of Northern and Southern routes"},
            {"lat":39.65,"lon":66.96,"label":"Samarkand","color":"#ffb800","info":"Greatest city of Central Asia"},
            {"lat":37.63,"lon":62.18,"label":"Merv","color":"#ffb800","info":"Major oasis city — gateway to Persia"},
            {"lat":38.08,"lon":46.29,"label":"Tabriz","color":"#ffb800","info":"Key Persian trading hub"},
            {"lat":41.01,"lon":28.97,"label":"Constantinople","color":"#00bfff","info":"Byzantine capital — western terminus"},
            {"lat":33.51,"lon":36.29,"label":"Damascus","color":"#ffb800","info":"Southern route Mediterranean gateway"},
            {"lat":31.20,"lon":29.92,"label":"Alexandria","color":"#00bfff","info":"Mediterranean port terminus"},
            {"lat":41.90,"lon":12.50,"label":"Rome","color":"#00bfff","info":"Final western destination"},
        ],
    },
    "belt and road": {
        "flyto": {"lat": 35.0, "lon": 80.0, "zoom": 3},
        "routes": [
            {"label": "China–Central Asia–Europe Corridor", "color": "#00ff66", "width": 2, "points": [
                [39.92,116.39],[36.06,103.83],[43.65,51.18],[41.30,69.24],
                [39.65,66.96],[37.95,58.38],[35.69,51.39],[41.01,28.97],
                [48.14,17.11],[52.52,13.41],[48.86,2.35],
            ]},
            {"label": "Maritime Road", "color": "#00bfff", "width": 2, "points": [
                [22.27,114.17],[1.35,103.82],[6.93,79.85],[11.59,43.14],
                [-4.05,39.67],[31.20,29.92],[41.90,12.50],[38.72,-9.14],[51.50,-0.12],
            ]},
        ],
    },
    "trans-siberian": {
        "flyto": {"lat": 57.0, "lon": 80.0, "zoom": 3},
        "routes": [
            {"label": "Trans-Siberian Railway", "color": "#ff9800", "width": 2, "points": [
                [55.75,37.62],[56.85,60.61],[57.15,65.53],[55.03,73.37],
                [53.36,83.75],[56.49,84.97],[56.01,92.87],[52.29,104.30],
                [51.83,107.61],[52.27,104.30],[53.73,127.53],[48.48,135.08],
                [43.12,131.90],
            ]},
        ],
    },
}

def _match_known_route(query: str) -> Optional[Dict]:
    q = query.lower()
    for key, data in KNOWN_ROUTES.items():
        if key in q:
            return data
    return None

# ── Gemini ARIA co-pilot ──────────────────────────────────────────
_ARIA_SYSTEM = (
    "You are ARIA (Analytical Reconnaissance Intelligence Assistant), "
    "the AI co-pilot embedded in EventNine — a Global Situational Awareness Platform. "
    "You have DIRECT CONTROL over the tactical map canvas. Use it for every response.\n\n"
    "Specializations: maritime domain awareness, aerospace tracking, conflict zone monitoring, "
    "cyber threats, SIGINT, weather systems, global trade flows, geopolitical analysis.\n\n"
    "ALWAYS respond with this exact JSON structure — no exceptions:\n"
    "{\"reply\": \"intelligence text\", \"map\": MAP_OBJECT_OR_NULL}\n\n"
    "Map object fields (use whichever apply):\n"
    "  flyto: {\"lat\": N, \"lon\": N, \"zoom\": N}\n"
    "  markers: [{\"lat\": N, \"lon\": N, \"label\": \"str\", \"color\": \"#hex\", \"info\": \"str\"}]\n"
    "  routes: [{\"points\": [[lat,lon],...], \"color\": \"#hex\", \"label\": \"str\", \"width\": 2}] "
    "— routes are rendered geodesic (great-circle), waypoints are interpolated automatically\n"
    "  weather_markers: [{\"lat\": N, \"lon\": N, \"label\": \"city\", \"temp\": \"24°C\", \"condition\": \"str\"}]\n"
    "  circles: [{\"lat\": N, \"lon\": N, \"radius\": meters_int, \"color\": \"#hex\", \"label\": \"str\"}]\n"
    "  sectors: [{\"lat\": N, \"lon\": N, \"radius_km\": N, \"bearing_start\": 0-360, \"bearing_end\": 0-360, \"color\": \"#hex\", \"label\": \"str\"}] "
    "— USE sectors (NOT circles) for radar coverage, missile threat arcs, air defense wedges, sonar cones. "
    "bearing_start/end define the exact angular coverage (e.g. S-400 facing west: bearing_start:240, bearing_end:300)\n\n"
    "Strict rules:\n"
    "- locate/find/show/where is X → flyto exact coordinates + marker at that spot\n"
    "- historical/trade/shipping routes → use MULTIPLE route entries for each branch/arm, "
    "  each with 10-20 precise waypoints following real geography through actual cities and passes. "
    "  For example the Silk Road has a Northern Route AND a Southern Route — draw BOTH as separate route entries with different colors.\n"
    "- weather/temperature of city → weather_markers with real approximate temp data\n"
    "- conflict zones → markers at key positions, circles for controlled areas\n"
    "- map is null ONLY for abstract/conceptual questions with no geographic component\n"
    "- reply: under 200 words, **bold** key entities. No speculation beyond known data.\n"
    "- label fields in map objects must be plain text only — NO markdown, NO asterisks, NO backticks.\n"
    "CRITICAL for routes: Use minimum 10 waypoints per route. Follow real geography — "
    "go through actual mountain passes, river valleys, coastal paths. "
    "Use accurate real-world lat/lon for every named city/waypoint."
)
_aria_client = genai_sdk.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

# ── Groq (primary — 14,400 req/day free) ─────────────────────────
from groq import Groq as _Groq
_groq_client = _Groq(api_key=os.getenv("GROQ_API_KEY", ""))
_GROQ_MODEL  = "llama-3.3-70b-versatile"

app = FastAPI(title="EventNine — Global Situational Awareness Dashboard", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

geo_processor  = GeometryProcessor()
fetcher        = UniversalFetcher()
fleet_sim      = FleetSimulator()


# ─────────────────────────────────────────────────────────────────
# REST
# ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.0.0", "timestamp": time.time()}

@app.post("/api/orchestrate")
def api_orchestrate(req: QueryRequest):
    try:
        return route_intent(req.prompt)
    except Exception as e:
        raise HTTPException(500, str(e))

_ARIA_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]

async def _groq_chat(msg: str) -> dict:
    """Call Groq (Llama 3.3 70B). Returns {reply, map} dict."""
    resp = await asyncio.to_thread(
        _groq_client.chat.completions.create,
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": _ARIA_SYSTEM},
            {"role": "user",   "content": msg},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip()
    parsed = json.loads(raw)
    return {"reply": parsed.get("reply", raw), "map": parsed.get("map")}


async def _gemini_chat(msg: str) -> dict:
    """Gemini fallback. Returns {reply, map} dict."""
    from google.genai import types as genai_types
    cfg = genai_types.GenerateContentConfig(
        system_instruction=_ARIA_SYSTEM,
        max_output_tokens=1000,
        response_mime_type="application/json",
    )
    for model in _ARIA_MODELS:
        try:
            response = await asyncio.to_thread(
                _aria_client.models.generate_content,
                model=model, contents=msg, config=cfg,
            )
            raw = response.text.strip()
            try:
                parsed = json.loads(raw)
                return {"reply": parsed.get("reply", raw), "map": parsed.get("map")}
            except json.JSONDecodeError:
                return {"reply": raw, "map": None}
        except Exception as e:
            err_s = str(e)
            if "RESOURCE_EXHAUSTED" in err_s or "429" in err_s:
                continue
            raise
    raise Exception("quota")


# ── Real weather from Open-Meteo (free, no key) ───────────────────
_WMO = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    45:"Fog",48:"Icy fog",
    51:"Light Drizzle",53:"Drizzle",55:"Heavy Drizzle",
    56:"Freezing Drizzle",57:"Heavy Freezing Drizzle",
    61:"Light Rain",63:"Rain",65:"Heavy Rain",
    66:"Freezing Rain",67:"Heavy Freezing Rain",
    71:"Light Snow",73:"Snow",75:"Heavy Snow",77:"Snow grains",
    80:"Rain Shower",81:"Rain Showers",82:"Violent Rain Shower",
    85:"Snow Shower",86:"Heavy Snow Shower",
    95:"Thunderstorm",96:"Thunderstorm with Hail",99:"Thunderstorm with Hail",
}

async def _fetch_real_weather(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weathercode&temperature_unit=celsius&timezone=auto"
    )
    async with httpx.AsyncClient(timeout=6.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    current = data.get("current", {})
    temp_c = current.get("temperature_2m")
    code   = current.get("weathercode", 0)
    return {
        "temp": f"{round(temp_c)}°C" if temp_c is not None else "?°C",
        "condition": _WMO.get(int(code), "Unknown"),
    }

async def _enrich_weather_markers(markers: list) -> list:
    async def _enrich_one(m):
        try:
            real = await _fetch_real_weather(m["lat"], m["lon"])
            m["temp"] = real["temp"]
            m["condition"] = real["condition"]
        except Exception as e:
            logger.warning(f"Weather fetch failed for {m.get('label')}: {e}")
        return m
    return list(await asyncio.gather(*[_enrich_one(m) for m in markers]))


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    msg = req.message
    if req.context:
        msg += f"\n\nContext: {req.context}"

    # Check hardcoded routes first — AI hallucinates lat/lon for known geography
    known_map = _match_known_route(msg)

    # Get AI text (and AI map for non-hardcoded queries)
    ai_result = None
    try:
        ai_result = await _groq_chat(msg)
    except Exception as groq_err:
        logger.warning(f"Groq failed: {groq_err} — falling back to Gemini")
        try:
            ai_result = await _gemini_chat(msg)
        except Exception:
            pass

    if ai_result is None:
        return {
            "reply": (
                "**ARIA offline** — All AI providers have reached their quota.\n\n"
                "The map and live data layers continue operating normally. "
                "ARIA will be available again after quota resets (midnight Pacific time)."
            ),
            "ok": False,
            "map": known_map,
        }

    # Use hardcoded map if available, otherwise use AI-generated map
    final_map = known_map if known_map else ai_result.get("map")

    # Replace AI-hallucinated weather with real Open-Meteo data
    if final_map and final_map.get("weather_markers"):
        final_map["weather_markers"] = await _enrich_weather_markers(final_map["weather_markers"])

    return {
        "reply": ai_result["reply"],
        "ok": True,
        "provider": ai_result.get("provider", "groq"),
        "map": final_map,
    }

class ReportRequest(BaseModel):
    query: str

@app.post("/api/report")
async def api_report(req: ReportRequest):
    from google.genai import types as genai_types
    prompt = (
        f'ARIA tactical intelligence AI. Generate a report for: "{req.query}". '
        f'Return JSON with these fields: topic(str), region(str), flyto(str: one of middle east/ukraine/asia/europe/iran/israel/china/russia/africa/usa), '
        f'kpis(array of 4: label/value/delta/status[live|warning|critical]), '
        f'activity_timeline(labels[7 dates]/series[2 items: name+data[7 nums]]), '
        f'threat_distribution(labels[5]/values[5 nums]), '
        f'signal_metrics(labels[Mon-Sun]/series[2: name+data[7 nums]]), '
        f'entity_breakdown(labels[5]/values[5 nums]), '
        f'system_status(array of 4: label/value[0-100 int]), '
        f'event_log(array of 8: time/event/domain/severity[HIGH|MEDIUM|LOW]), '
        f'intel_items(array of 3: title/content), '
        f'intel_summary(str). Use realistic data specific to the topic.'
    )
    # Try Groq first
    try:
        resp = await asyncio.to_thread(
            _groq_client.chat.completions.create,
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.5,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        return {"ok": True, "data": data, "provider": "groq"}
    except Exception as groq_err:
        logger.warning(f"Groq report failed: {groq_err} — trying Gemini")

    # Gemini fallback
    from google.genai import types as genai_types
    cfg = genai_types.GenerateContentConfig(
        max_output_tokens=2000,
        response_mime_type="application/json",
    )
    for model in _ARIA_MODELS:
        try:
            response = await asyncio.to_thread(
                _aria_client.models.generate_content,
                model=model, contents=prompt, config=cfg,
            )
            raw = response.text.strip()
            if raw.startswith('```'):
                raw = raw[raw.index('\n')+1:] if '\n' in raw else raw[3:]
            if raw.endswith('```'):
                raw = raw[:raw.rfind('```')]
            data = json.loads(raw.strip())
            return {"ok": True, "data": data, "provider": model}
        except json.JSONDecodeError:
            continue
        except Exception as e:
            err_s = str(e)
            if "RESOURCE_EXHAUSTED" in err_s or "429" in err_s:
                continue
            break
    raise HTTPException(503, "All AI providers exhausted or failed")

@app.post("/api/simulation/build")
async def api_build(req: QueryRequest):
    try:
        intent = route_intent(req.prompt)
        if intent.mode == IntentMode.LIVE_TRACKING and intent.fetch_plan:
            entities = await fetcher.fetch(intent.fetch_plan)
            return {"mode": "live_tracking", "label": intent.display_label,
                    "entities": [e.model_dump() for e in entities]}
        if intent.asset_intent:
            payload = await geo_processor.build_3d_simulation_payload(intent.asset_intent)
            return payload
        return {"error": "unresolvable intent"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connected  total={len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WS disconnected total={len(self.active)}")

manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────
# 3-D model frame helpers (unchanged from v2)
# ─────────────────────────────────────────────────────────────────

def _mechanical_frame(components, explosion_factor, rotation_angle):
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
    Ry = np.array([[cos_a,0,sin_a],[0,1,0],[-sin_a,0,cos_a]])
    frame = {}
    for comp in components:
        verts   = np.array(comp["vertices"])
        exp_vec = np.array(comp.get("explosion_vector", [0,1,0]))
        T       = np.array(comp["global_transform"]).reshape(4,4)
        lr, lt  = T[:3,:3], T[:3,3]
        out     = (verts @ lr.T) + lt
        if explosion_factor > 1e-4:
            out += exp_vec * explosion_factor * 1.5
        pt = comp.get("primitive_type","").lower()
        if pt in ("gear","shaft","wheel","cylinder","rotor"):
            spin = rotation_angle * (2.0 if pt == "gear" else 1.0)
            cs,ss = np.cos(spin), np.sin(spin)
            Rz  = np.array([[cs,-ss,0],[ss,cs,0],[0,0,1]])
            out = (verts @ lr.T) @ Rz.T + lt
            if explosion_factor > 1e-4:
                out += exp_vec * explosion_factor * 1.5
        frame[comp["name"]] = (out @ Ry.T).tolist()
    return frame


def _telemetry_frame(components, scene_traj, path_t, explosion_factor, idle_rot):
    path_M = geo_processor.compute_path_transform(scene_traj, path_t, idle_rot)
    rot3, t3 = path_M[:3,:3], path_M[:3,3]
    coords = {}
    for comp in components:
        verts   = np.array(comp["vertices"])
        exp_vec = np.array(comp.get("explosion_vector",[0,1,0]))
        lt_mat  = np.array(comp["global_transform"]).reshape(4,4)
        lv      = (verts @ lt_mat[:3,:3].T) + lt_mat[:3,3]
        world   = (lv @ rot3.T) + t3
        if explosion_factor > 1e-4:
            world = ((lv + exp_vec * explosion_factor * 0.6) @ rot3.T) + t3
        coords[comp["name"]] = world.tolist()
    return {"path_transform": path_M.flatten().tolist(), "coords": coords}


# ─────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    """
    Unified WebSocket for both live-tracking layers and 3-D model streaming.

    ── Client → Server ────────────────────────────────────────────
    { "type": "query",        "prompt": str }   → live tracking layer
    { "type": "initialize",   "prompt": str }   → 3-D model mode
    { "type": "remove_layer", "layer_id": str } → stop & remove a layer
    { "type": "clear_all" }                     → clear everything
    { "type": "control", ... }                  → existing 3-D controls

    ── Server → Client ────────────────────────────────────────────
    { "type": "layer_init",    layer_id, label, source, model_key, color }
    { "type": "entity_update", layer_id, entities: [...] }
    { "type": "layer_removed", layer_id }
    { "type": "mesh_init",     ... }             (3-D mode)
    { "type": "vertex_update", ... }             (3-D mode 30 Hz)
    { "type": "error",         message }
    """
    await manager.connect(ws)

    # Per-connection state
    state: Dict[str, Any] = {
        # Live-tracking layers
        "layers":         {},   # layer_id → {"plan": DataFetchPlan, "task": Task}

        # 3-D model state
        "active_mesh":    None,
        "asset_mode":     None,
        "scene_traj":     [],
        "is_animating":   False,
        "rotation_speed": 0.5,
        "rotation_angle": 0.0,
        "animate_explode":False,
        "explosion_factor": 0.0,
        "explosion_dir":  1.0,
        "path_t":         0.0,
        "path_speed":     0.04,
        "path_loop":      True,
    }

    # ── Live-tracking layer refresh task ──────────────────────────

    async def layer_loop(layer_id: str, plan: DataFetchPlan):
        """Fetches data at plan.refresh_s interval and pushes to client."""
        while layer_id in state["layers"]:
            try:
                entities: List[LiveEntity] = await fetcher.fetch(plan)
                if layer_id not in state["layers"]:
                    break
                await ws.send_json({
                    "type":     "entity_update",
                    "layer_id": layer_id,
                    "count":    len(entities),
                    "entities": [e.model_dump() for e in entities],
                })
                logger.info(f"Layer {layer_id[:8]} pushed {len(entities)} entities")
            except Exception as e:
                logger.error(f"Layer {layer_id[:8]} fetch error: {e}")
            await asyncio.sleep(plan.refresh_s)

    # ── 3-D model 30-Hz broadcast loop ───────────────────────────

    async def model_broadcast_loop():
        last = time.time()
        while True:
            if state["active_mesh"] is None:
                await asyncio.sleep(0.1)
                continue
            now = time.time()
            dt  = now - last; last = now

            if state["is_animating"]:
                state["rotation_angle"] += state["rotation_speed"] * dt * 2.0

            if state["animate_explode"]:
                state["explosion_factor"] += state["explosion_dir"] * dt * 0.4
                if state["explosion_factor"] >= 1.0:
                    state["explosion_factor"] = 1.0; state["explosion_dir"] = -1.0
                elif state["explosion_factor"] <= 0.0:
                    state["explosion_factor"] = 0.0; state["explosion_dir"] =  1.0

            try:
                comps = state["active_mesh"]["components"]
                mode  = state["asset_mode"]

                if mode == "telemetry" and state["scene_traj"]:
                    if state["is_animating"]:
                        state["path_t"] += state["path_speed"] * dt
                        if state["path_t"] >= 1.0:
                            state["path_t"] = 0.0 if state["path_loop"] else 1.0
                    frame = _telemetry_frame(comps, state["scene_traj"],
                                             state["path_t"], state["explosion_factor"],
                                             state["rotation_angle"] * 0.15)
                    await ws.send_json({
                        "type": "vertex_update", "mode": "telemetry",
                        "timestamp": time.time(), "path_t": state["path_t"],
                        "explosion_factor": state["explosion_factor"],
                        "path_transform": frame["path_transform"],
                        "coords": frame["coords"],
                    })
                else:
                    coords = _mechanical_frame(comps, state["explosion_factor"],
                                               state["rotation_angle"])
                    await ws.send_json({
                        "type": "vertex_update", "mode": mode or "mechanical",
                        "timestamp": time.time(),
                        "explosion_factor": state["explosion_factor"],
                        "rotation_angle": state["rotation_angle"],
                        "coords": coords,
                    })
            except Exception as e:
                logger.error(f"3-D frame error: {e}")
                break

            await asyncio.sleep(0.033)

    model_task = asyncio.create_task(model_broadcast_loop())

    # ── Command receive loop ──────────────────────────────────────

    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except Exception:
                await ws.send_json({"type":"error","message":"Invalid JSON"})
                continue

            t = cmd.get("type")

            # ── QUERY — live tracking layer ──────────────────────
            if t == "query":
                prompt = cmd.get("prompt","").strip()
                if not prompt:
                    continue

                intent = route_intent(prompt)

                if intent.mode == IntentMode.LIVE_TRACKING and intent.fetch_plan:
                    plan      = intent.fetch_plan
                    layer_id  = str(uuid.uuid4())

                    await ws.send_json({
                        "type":        "layer_init",
                        "layer_id":    layer_id,
                        "label":       intent.display_label,
                        "source":      plan.source,
                        "model_key":   plan.model_key,
                        "color":       plan.entity_color,
                        "refresh_s":   plan.refresh_s,
                    })

                    task = asyncio.create_task(layer_loop(layer_id, plan))
                    state["layers"][layer_id] = {"plan": plan, "task": task}
                    logger.info(f"Layer created: {layer_id[:8]} [{plan.source}]")

                else:
                    # Not a live-tracking query — treat as 3-D model init
                    cmd["type"] = "initialize"
                    cmd["prompt"] = prompt
                    # fall through to initialize handler below
                    t = "initialize"

            # ── INITIALIZE — 3-D model ───────────────────────────
            if t == "initialize":
                prompt = cmd.get("prompt","mechanical piston assembly")
                intent = route_intent(prompt)

                ai: Optional[AssetIntent] = intent.asset_intent
                if ai is None:
                    # Wrap a minimal AssetIntent if Gemini returned a live plan for this
                    await ws.send_json({"type":"error","message":"Use the query command for live tracking."})
                    continue

                payload = await geo_processor.build_3d_simulation_payload(ai)
                state["active_mesh"]  = payload
                state["asset_mode"]   = payload["asset_type"]
                state["is_animating"] = True
                state["path_t"]       = 0.0

                tel = payload.get("telemetry")
                state["scene_traj"] = tel["scene_trajectory"] if tel else []

                await ws.send_json({
                    "type":       "mesh_init",
                    "asset_name": payload["asset_name"],
                    "asset_type": payload["asset_type"],
                    "components": [
                        {"name": c["name"], "primitive_type": c["primitive_type"],
                         "faces": c["faces"], "color": c["color"],
                         "wireframe_edges": c.get("wireframe_edges",[])}
                        for c in payload["components"]
                    ],
                    "telemetry": tel,
                })

            # ── REMOVE LAYER ─────────────────────────────────────
            elif t == "remove_layer":
                lid = cmd.get("layer_id")
                if lid in state["layers"]:
                    state["layers"][lid]["task"].cancel()
                    del state["layers"][lid]
                    await ws.send_json({"type":"layer_removed","layer_id":lid})

            # ── CLEAR ALL ────────────────────────────────────────
            elif t == "clear_all":
                for lid, entry in list(state["layers"].items()):
                    entry["task"].cancel()
                state["layers"].clear()
                state["active_mesh"] = None
                await ws.send_json({"type":"cleared"})

            # ── CONTROL ─────────────────────────────────────────
            elif t == "control":
                for k in ("is_animating","animate_explosion","path_loop"):
                    if k in cmd:
                        state[k.replace("animate_explosion","animate_explode")] = bool(cmd[k])
                if "animate_explosion" in cmd:
                    state["animate_explode"] = bool(cmd["animate_explosion"])
                if "explosion_factor" in cmd:
                    state["explosion_factor"] = float(cmd["explosion_factor"])
                    state["animate_explode"]  = False
                if "rotation_speed" in cmd:
                    state["rotation_speed"] = float(cmd["rotation_speed"])
                if "path_speed" in cmd:
                    state["path_speed"] = float(cmd["path_speed"])
                if "path_t" in cmd:
                    state["path_t"] = float(cmd["path_t"])
                if "path_loop" in cmd:
                    state["path_loop"] = bool(cmd["path_loop"])

    except WebSocketDisconnect:
        manager.disconnect(ws)
    finally:
        model_task.cancel()
        for entry in state["layers"].values():
            entry["task"].cancel()
        logger.info("WS session closed, all tasks cancelled.")


# ─────────────────────────────────────────────────────────────────
# /ws/operations-stream — 30 Hz intelligence broadcast
# ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/operations-stream")
async def ws_ops_stream(ws: WebSocket):
    await ws.accept()
    logger.info("OPS stream connected")
    try:
        while True:
            raw = None
            # Non-blocking check for incoming command
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            if raw:
                try:
                    cmd = json.loads(raw)
                    if cmd.get("type") == "activate":
                        intent = parse_operational_intent(cmd.get("prompt", "global operations"))
                        fleet_sim.activate(
                            theater     = intent["theater"],
                            domains     = intent["domains"],
                            threat_level= intent["threat_level"],
                            bbox        = intent["bbox"],
                            integrity   = intent["integrity"],
                        )
                        await ws.send_json({
                            "type":    "activated",
                            "theater": intent["theater"],
                            "domains": intent["domains"],
                            "threat":  intent["threat_level"],
                            "bbox":    intent["bbox"],
                        })
                    elif cmd.get("type") == "deactivate":
                        fleet_sim.deactivate()
                        await ws.send_json({"type": "deactivated"})
                except Exception as e:
                    logger.error(f"OPS cmd error: {e}")

            if fleet_sim.is_active:
                try:
                    payload = fleet_sim.tick()
                    await ws.send_json({"type": "intel_update", "payload": payload})
                except Exception as e:
                    logger.error(f"OPS tick error: {e}")

            await asyncio.sleep(1 / 30)

    except WebSocketDisconnect:
        logger.info("OPS stream disconnected")
    except Exception as e:
        logger.error(f"OPS stream error: {e}")


# ─────────────────────────────────────────────────────────────────
# /ws/terminal-logs — defense-grade log stream
# ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/terminal-logs")
async def ws_terminal_logs(ws: WebSocket):
    await ws.accept()
    logger.info("Terminal log stream connected")
    try:
        while True:
            threat = "OMEGA_CRITICAL" if fleet_sim.is_active and fleet_sim._threat == "OMEGA_CRITICAL" \
                     else ("BRAVO_MONITORED" if fleet_sim.is_active else "ALPHA_CLEAR")
            node_count = sum(len(v) for v in fleet_sim._fleets.values()) if fleet_sim.is_active else 0
            log_line = generate_terminal_log(node_count, threat)
            await ws.send_json({
                "type":      "log",
                "message":   log_line,
                "level":     "CRITICAL" if "OMEGA" in threat else ("WARN" if "BRAVO" in threat else "INFO"),
                "timestamp": time.time(),
            })
            await asyncio.sleep(0.4 + 0.6 * (threat == "ALPHA_CLEAR"))
    except WebSocketDisconnect:
        logger.info("Terminal log stream disconnected")
    except Exception as e:
        logger.error(f"Terminal log error: {e}")


# ─────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    try:
        return HTMLResponse(open("index.html", encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse("<h3>EventNine v3 running. index.html not found.</h3>")
