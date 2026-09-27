"""Run the API tests against Tiger Data (TimescaleDB) as well as SQLite.

    PWW_TEST_DATABASE_URL=postgres://postgres:pw@127.0.0.1:5432/postgres python -m pytest

With that set, every test gets its own fresh database on that server. Without it, tests use SQLite.
"""
import os
import uuid

import pytest

TEST_DATABASE_URL = os.getenv("PWW_TEST_DATABASE_URL", "")


@pytest.fixture(autouse=True)
def database(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "RIVER_SYNC_MINUTES", 0)
    if not TEST_DATABASE_URL:
        monkeypatch.setattr(config, "DATABASE_URL", "")
        yield ""
        return

    import psycopg
    from psycopg.conninfo import make_conninfo
    name = f"pww_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    url = make_conninfo(TEST_DATABASE_URL, dbname=name)
    monkeypatch.setattr(config, "DATABASE_URL", url)
    yield url
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


requires_tiger = pytest.mark.skipif(not TEST_DATABASE_URL, reason="set PWW_TEST_DATABASE_URL to a TimescaleDB server")
