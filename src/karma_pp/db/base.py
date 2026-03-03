# db/base.py
import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import structlog

from . import DB_REPOS

log = structlog.get_logger(__name__)

# locate project root / data/karma.db
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "karma.db"


def connect() -> duckdb.DuckDBPyConnection:
    """
    Open (or create) the DuckDB file, apply all schema.sql under db/*/,
    and return a fresh connection.
    """
    log.info("db-connect-start", db_path=str(DB_PATH))
    conn = duckdb.connect(str(DB_PATH), read_only=False)

    # Check if database is new by looking for expected tables
    tables_result = conn.execute("SHOW TABLES").fetchall()
    existing_tables = {row[0] for row in tables_result}
    expected_tables = set(DB_REPOS.keys())

    if not expected_tables.issubset(existing_tables):
        # Database is new or missing tables, load schemas
        _load_schemas(conn)
    else:
        log.debug("db-schemas-exist")

    log.info(
        "db-connect-complete", db_path=str(DB_PATH), num_tables=len(existing_tables)
    )
    return conn


def _load_schemas(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Load all database schemas in dependency order.
    """
    base = Path(__file__).parent
    num_schemas = len(DB_REPOS)

    log.info("db-load-schemas-start", num_schemas=num_schemas, base_dir=str(base))

    # Load schemas using DB_REPOS configuration
    for repo_name in DB_REPOS:
        schema_file = f"{repo_name}/schema.sql"
        schema_path = base / schema_file
        if schema_path.exists():
            log.debug(
                "db-load-schema", repo_name=repo_name, schema_file=str(schema_path)
            )
            sql = schema_path.read_text(encoding="utf-8")
            conn.execute(sql)
            log.debug(
                "db-schema-loaded", repo_name=repo_name, schema_file=str(schema_path)
            )
        else:
            log.warning("db-schema-missing", schema_file=str(schema_path))

    # Get all table names from the database
    tables_result = conn.execute("SHOW TABLES").fetchall()
    table_names = [row[0] for row in tables_result]

    log.info("db-load-schemas-complete", tables=table_names)


class BaseRepo[RM]:
    """
    Generic repository for a dataclass-backed table.
    Your model must be a @dataclass with:
      __table__ = "<table_name>"
      __pk__    = "<primary_key_column>"
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, model: RM):
        assert is_dataclass(model), "model must be a dataclass"
        self.conn = conn
        self.model = model
        self.table = getattr(model, "__table__", model.__name__.lower())
        self.pk = getattr(model, "__pk__", None)
        self.cols = [f.name for f in fields(model)]

    def insert(self, obj: RM, replace: bool = True):
        verb = "OR REPLACE" if replace else ""
        cols = ", ".join(self.cols)
        ph = ", ".join("?" for _ in self.cols)
        sql = f"INSERT {verb} INTO {self.table} ({cols}) VALUES ({ph})"
        vals = [getattr(obj, c) for c in self.cols]
        self.conn.execute(sql, vals)

    def get(self, key: any) -> Optional[RM]:
        if not self.pk:
            raise ValueError("Primary key (__pk__) not set on model")
        cols = ", ".join(self.cols)
        sql = f"SELECT {cols} FROM {self.table} WHERE {self.pk} = ?"
        row = self.conn.execute(sql, (key,)).fetchone()
        return self.model(*row) if row else None

    def list_all(self) -> list[RM]:
        cols = ", ".join(self.cols)
        rows = self.conn.execute(f"SELECT {cols} FROM {self.table}").fetchall()
        return [self.model(*r) for r in rows]

    def filter_by(self, **kwargs) -> list[RM]:
        cols = ", ".join(self.cols)
        clauses = " AND ".join(f"{k} = ?" for k in kwargs)
        vals = tuple(kwargs.values())
        sql = f"SELECT {cols} FROM {self.table} WHERE {clauses}"
        rows = self.conn.execute(sql, vals).fetchall()
        return [self.model(*r) for r in rows]


class HashedJSONRepo[RM](BaseRepo[RM]):
    """
    Content-addressable repository mix-in.

    Each concrete repo stores a component (game, population, …) as
    *one* canonical JSON string plus a SHA-256 primary-key column
    ``<hash_col>``.  The mix-in provides:

    * ``ensure(blob)`` – upsert the row and return the 64-char hash.
    * ``fetch(hash)`` – retrieve the dataclass row or raise *KeyError*.

    Subclasses must set ``hash_col`` (e.g. ``"game_hash"``).  They may
    override ``_canonicalize()`` to inject defaults or reorder lists
    before hashing.
    """

    # must be overridden
    hash_col: str

    @staticmethod
    def _canonicalize(blob: dict[str, any]) -> str:
        """Return JSON with sorted keys and no insignificant whitespace."""
        return json.dumps(blob, sort_keys=True, separators=(",", ":"))

    def ensure(self, blob: dict[str, any]) -> str:
        """
        Insert the component if it is new and return its hash.

        Parameters
        ----------
        blob : dict
            Raw config for the component.

        Returns
        -------
        str
            64-character SHA-256 of the canonical JSON.
        """
        canonical_json = self._canonicalize(blob)  # now JSON
        h = hashlib.sha256(canonical_json.encode()).hexdigest()

        self.conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.table}
              ({self.hash_col}, json, created_at)
            VALUES (?, ?, ?)
            """,
            (h, canonical_json, datetime.utcnow()),
        )
        return h

    def fetch(self, h: str) -> RM:
        """
        Retrieve a row by its hash or raise *KeyError*.
        """
        row = self.get(h)
        if not row:
            raise KeyError(f"{self.table}: unknown hash {h}")
        return row


class Database:
    """
    Database connection with explicitly configured repositories.
    Exposes repos as attributes: db.world, db.population, db.mechanism,
    db.experiment, db.metric, db.artifact, db.exp_group
    """

    def __init__(self):
        log.info("db-init-start")
        self.conn = connect()
        self._setup_repos()
        log.info("db-init-complete")

    def _setup_repos(self):
        """Set up all database repositories using DB_REPOS configuration."""
        num_repos = len(DB_REPOS)
        base = Path(__file__).parent

        log.info("db-setup-repos-start", num_repos=num_repos, base_dir=str(base))

        # Create repo instances using DB_REPOS configuration
        for repo_name, (module_path, class_name) in DB_REPOS.items():
            module = __import__(f"karma_pp.db.{module_path}", fromlist=[class_name])
            repo_class = getattr(module, class_name)
            setattr(self, repo_name, repo_class(self.conn))
            log.debug("db-repo-loaded", repo_name=repo_name, repo_class=class_name)

        log.info("db-setup-repos-complete", repos=list(DB_REPOS.keys()))

    def close(self):
        log.info("db-close")
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
