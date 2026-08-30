"""Versioned game and route profile catalog."""

import json
from pathlib import Path
from typing import Optional


PROFILE_SCHEMA_VERSION = "1.0"


class ProfileCatalog:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (
            Path(root)
            if root
            else Path(__file__).resolve().parents[3] / "config"
        )
        self.games = self._load("games", "profile_id")
        self.routes = self._load("routes", "route_profile_id")
        for route in self.routes.values():
            if route["game_profile_id"] not in self.games:
                raise RuntimeError(
                    "Route profile {} references unknown game profile {}".format(
                        route["route_profile_id"], route["game_profile_id"]
                    )
                )

    def _load(self, directory: str, identifier: str) -> dict:
        values = {}
        path = self.root / directory
        if not path.exists():
            return values
        for filename in sorted(path.glob("*.json")):
            value = json.loads(filename.read_text(encoding="utf-8"))
            if value.get("schema_version") != PROFILE_SCHEMA_VERSION:
                raise RuntimeError("Unsupported profile schema in {}".format(filename))
            profile_id = value.get(identifier)
            if not profile_id or not isinstance(profile_id, str):
                raise RuntimeError("Profile {} has no {}".format(filename, identifier))
            if profile_id in values:
                raise RuntimeError("Duplicate profile ID: {}".format(profile_id))
            copied = dict(value)
            copied["source_file"] = str(filename)
            values[profile_id] = copied
        return values

    def game(self, profile_id: str) -> dict:
        try:
            return self.games[profile_id]
        except KeyError:
            raise KeyError("Unknown game profile: {}".format(profile_id))

    def route(self, profile_id: str) -> dict:
        try:
            return self.routes[profile_id]
        except KeyError:
            raise KeyError("Unknown route profile: {}".format(profile_id))

    def descriptor(self) -> dict:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "games": list(self.games.values()),
            "routes": list(self.routes.values()),
        }
