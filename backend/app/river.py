"""Live river conditions from USGS gauges on Philadelphia waterways, stored in Tiger Data.

USGS publishes readings roughly every 15 minutes (provisional data). Storing them next to the litter
reports lets the dashboard compare litter with river flow, e.g. whether reports rise after storms
push water and trash out of storm drains and combined sewer overflows.

Run a backfill once:      python -m app.river --days 30
The API also polls every PWW_RIVER_SYNC_MINUTES (default 15) while it runs on Tiger Data.
"""
from __future__ import annotations

import argparse
import logging
import threading
from datetime import datetime

import httpx

from . import config

log = logging.getLogger("pww.river")

# USGS site number -> (name, waterway as used in reports). Edit PWW_RIVER_SITES to change which are polled.
SITES = {
    "01474500": ("Schuylkill River at Philadelphia", "Schuylkill River"),
    "01467200": ("Delaware River at Penn's Landing", "Delaware River"),
    "01474000": ("Wissahickon Creek at mouth", "Wissahickon Creek"),
    "01467048": ("Pennypack Creek at Lower Rhawn St", "Pennypack Creek"),
    "01475548": ("Cobbs Creek at Mt. Moriah Cemetery", "Cobbs Creek"),
    "01467087": ("Frankford Creek at Castor Ave", "Tacony-Frankford Creek"),
}

# USGS parameter code -> short name used in the database and API.
PARAMETERS = {
    "00060": "discharge_cfs",
    "00065": "gage_height_ft",
    "00010": "water_temp_c",
    "63680": "turbidity_fnu",
    "00300": "dissolved_oxygen_mg_l",
    "00095": "specific_conductance_us_cm",
}


def fetch(days: int, sites: list[str] | None = None, client: httpx.Client | None = None) -> dict:
    """Instantaneous values for the last `days` days (USGS allows up to 120)."""
    params = {
        "format": "json",
        "sites": ",".join(sites or config.RIVER_SITES or SITES),
        "parameterCd": ",".join(PARAMETERS),
        "period": f"P{max(1, min(days, 120))}D",
        "siteStatus": "all",
    }
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        response = client.get(config.USGS_IV_URL, params=params)
        response.raise_for_status()
        return response.json()
    finally:
        if own:
            client.close()


def parse(payload: dict) -> tuple[list[dict], list[tuple]]:
    """USGS WaterML-JSON -> (site rows, (time, site_id, parameter, value) readings)."""
    sites: dict[str, dict] = {}
    readings: list[tuple] = []
    for series in payload.get("value", {}).get("timeSeries", []):
        info = series.get("sourceInfo", {})
        try:
            site_id = info["siteCode"][0]["value"]
            code = series["variable"]["variableCode"][0]["value"]
        except (KeyError, IndexError, TypeError):
            continue
        parameter = PARAMETERS.get(code)
        if parameter is None:
            continue
        name, waterway = SITES.get(site_id, (info.get("siteName", site_id), None))
        geo = info.get("geoLocation", {}).get("geogLocation", {})
        sites[site_id] = {"site_id": site_id, "name": name, "waterway": waterway,
                          "latitude": geo.get("latitude"), "longitude": geo.get("longitude")}
        no_data = series["variable"].get("noDataValue")
        # A site can report the same parameter from several sensors; keep them all, the table dedupes by time.
        for block in series.get("values", []):
            for point in block.get("value", []):
                try:
                    value = float(point["value"])
                    when = datetime.fromisoformat(point["dateTime"])
                except (KeyError, TypeError, ValueError):
                    continue
                if no_data is not None and value == no_data:
                    continue
                readings.append((when, site_id, parameter, value))
    return list(sites.values()), readings


def sync(store, days: int = 1, client: httpx.Client | None = None) -> dict:
    """Fetch and store readings. Safe to repeat: readings already stored are skipped."""
    sites, readings = parse(fetch(days, client=client))
    store.upsert_river_sites(sites)
    added = store.insert_river_readings(readings)
    return {"sites": len(sites), "readings_fetched": len(readings), "readings_added": added}


class Poller:
    """Background thread that keeps river readings current while the API runs."""

    def __init__(self, store, minutes: float):
        self.store, self.minutes = store, minutes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="river-sync", daemon=True)
        self.last_result: dict | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        days = config.RIVER_BACKFILL_DAYS  # first run backfills, later runs only need the last day
        while not self._stop.is_set():
            try:
                self.last_result = sync(self.store, days)
                self.last_error = None
                days = 1
            except Exception as e:  # network or USGS outage: keep the API up and try again later
                self.last_error = str(e)
                log.warning("River sync failed: %s", e)
            self._stop.wait(self.minutes * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load USGS river readings into Tiger Data")
    parser.add_argument("--days", type=int, default=30, help="how many days to backfill (max 120)")
    args = parser.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("Set PWW_DATABASE_URL to your Tiger Data service first.")
    from .tigerstore import TigerStore
    store = TigerStore(config.DATA_DIR, config.DATABASE_URL)
    try:
        print(sync(store, args.days))
    finally:
        store.close()


if __name__ == "__main__":
    main()