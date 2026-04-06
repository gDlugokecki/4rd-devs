import json
import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from send_answer import send_answer

load_dotenv()

API_KEY = os.environ["AI_DEVS_4_API_KEY"]
HUB_BASE = "https://hub.ag3nts.org/api"
PROXIMITY_KM = 25.0

LOCATIONS_PATH = Path(__file__).parent / "findhim_locations.json"

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Zabrze": (50.3249, 18.7857),
    "Piotrków Trybunalski": (51.4052, 19.7028),
    "Grudziądz": (53.4836, 18.7536),
    "Tczew": (53.7786, 18.7759),
    "Radom": (51.4027, 21.1471),
    "Chelmno": (53.3498, 18.4257),
    "Żarnowiec": (54.5678, 18.2456),
}

DEFINITIONS = [
    {
        "name": "get_location",
        "description": "Fetch all known locations (list of lat/lon) where a person was seen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "surname": {"type": "string"},
            },
            "required": ["name", "surname"],
        },
    },
    {
        "name": "get_access_level",
        "description": "Fetch the access level of a person.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "surname": {"type": "string"},
                "birthYear": {"type": "string"},
            },
            "required": ["name", "surname", "birthYear"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer when a suspect near a power plant is identified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "surname": {"type": "string"},
                "accessLevel": {"type": "integer"},
                "powerPlant": {"type": "string", "description": "Power plant code, e.g. PWR1234PL"},
            },
            "required": ["name", "surname", "accessLevel", "powerPlant"],
        },
    },
    {
        "name": "find_nearby_power_plant",
        "description": (
            f"Check if coordinates are within {PROXIMITY_KM} km of a known power plant. "
            "Returns plant info or null."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            "required": ["lat", "lon"],
        },
    },
]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _load_power_plants() -> dict:
    with open(LOCATIONS_PATH) as f:
        return json.load(f)["power_plants"]


def get_location(name: str, surname: str) -> list[dict]:
    response = httpx.post(
        f"{HUB_BASE}/location",
        json={"apikey": API_KEY, "name": name, "surname": surname},
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json()


def get_access_level(name: str, surname: str, birthYear: str) -> dict:
    response = httpx.post(
        f"{HUB_BASE}/accesslevel",
        json={
            "apikey": API_KEY,
            "name": name,
            "surname": surname,
            "birthYear": birthYear,
        },
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json()


def find_nearby_power_plant(lat: float, lon: float) -> dict | None:
    for city, info in _load_power_plants().items():
        coords = CITY_COORDS.get(city)
        if coords is None:
            continue
        dist = _haversine(lat, lon, coords[0], coords[1])
        if dist <= PROXIMITY_KM:
            return {"city": city, "code": info["code"], "distance_km": round(dist, 2)}
    return None


def submit_answer(name: str, surname: str, access_level: int, power_plant: str) -> dict:
    return send_answer(
        task="findhim",
        answer={
            "name": name,
            "surname": surname,
            "accessLevel": access_level,
            "powerPlant": power_plant,
        },
    )


def process(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_location":
        result = get_location(tool_input["name"], tool_input["surname"])
    elif tool_name == "get_access_level":
        result = get_access_level(
            tool_input["name"], tool_input["surname"], tool_input["birthYear"]
        )
    elif tool_name == "find_nearby_power_plant":
        result = find_nearby_power_plant(tool_input["lat"], tool_input["lon"])
    elif tool_name == "submit_answer":
        result = submit_answer(
            tool_input["name"],
            tool_input["surname"],
            tool_input["accessLevel"],
            tool_input["powerPlant"],
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)
