"""The Circle's shared database against schema.sql: check it, apply what is missing, or see what it holds.

    python -m contexts.circle.schema --check    # read-only: what differs between the database and schema.sql
    python -m contexts.circle.schema --counts   # read-only: --check plus tables, columns, indexes, sizes, rows per circle
    python -m contexts.circle.schema --apply    # create the tables, indexes and views that are missing; nothing else

NEON_DATABASE_URL comes from the environment, else from `.env` (ATLAS_ENV_FILE). The connection string is never
printed, and a failed connection says only what kind of failure it was (the driver's own message names the host).

ATLAS_NEON_TRANSPORT picks the way there, as for the app: `tcp` (psycopg, port 5432), `https` (port 443, no
driver needed) or `auto` (the default: tcp, then https when 5432 cannot be reached). The first line printed says
which one was used. Over https, --check and --counts are still read-only (the server enforces it per request),
and --apply looks at what exists in one request and creates what is missing in a second, as one transaction.

--check and --counts open the connection read-only. They print shapes and counts: never row contents; member and
circle ids are shortened to two letters unless they are `test-…` rows. Exit status 0: the database matches.

--apply runs schema.sql's own statements, in one transaction, only for objects that do not exist yet. It cannot
alter, replace or drop anything: an existing table keeps its columns (a column schema.sql gained needs a written
migration, which --check reports), and an existing view is left as it is.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from contexts.circle.application.sharing import CircleOffline
from contexts.circle.infrastructure.neon import ENV_URL, NeonTransport
from contexts.circle.infrastructure.neon_https import NeonHttpsDropped

SCHEMA = Path(__file__).parent / "infrastructure" / "schema.sql"
TIME_COLUMNS = ("created_at", "joined_at", "shared_at", "updated_at", "played_at", "set_at", "closed_at", "day", "starts_at")


def expected(sql: str) -> dict:
    """The tables (with their columns), indexes and views schema.sql creates."""
    tables = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", sql, re.S):
        lines = [re.sub(r"--.*", "", line).strip() for line in body.splitlines()]
        tables[name] = [line.split()[0] for line in lines
                        if line and line.split()[0] not in ("CHECK", "PRIMARY", "UNIQUE", "FOREIGN", "CONSTRAINT")]
    return {"tables": tables, "indexes": dict(re.findall(r"CREATE INDEX IF NOT EXISTS (\w+) ON (\w+)", sql)),
            "views": re.findall(r"CREATE OR REPLACE VIEW (\w+)", sql)}


def short(value) -> str:
    text = str(value)
    return text if text.startswith("test-") else f"{text[:2]}…({len(text)})"


def url_from_env(env_file: str = ".env") -> str | None:
    if os.environ.get(ENV_URL):
        return os.environ[ENV_URL]
    try:
        lines = Path(os.environ.get("ATLAS_ENV_FILE", env_file)).read_text().splitlines()
    except OSError:
        return None
    return next((line.split("=", 1)[1].strip() for line in lines if line.startswith(f"{ENV_URL}=")), None) or None


def audit(conn, out=print, counts: bool = True) -> list[str]:
    """Print the report and return the mismatches with schema.sql (empty: the database matches the file).
    `counts=False` leaves out the per-table lines and prints the differences only."""
    quiet, say = (lambda *a: None), out
    if not counts:
        out = quiet
    cur = conn.cursor()

    def q(sql, params=()):
        cur.execute(sql, params)
        return cur.fetchall()

    want = expected(SCHEMA.read_text())
    out(f"server {q('SHOW server_version')[0][0]}, database size "
        f"{q('SELECT pg_size_pretty(pg_database_size(current_database()))')[0][0]}, "
        f"database clock minus this machine's: {q('SELECT extract(epoch FROM now()) - %s', (time.time(),))[0][0]:+.1f} s")
    have = {t: [] for (t,) in q("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                                " AND table_type = 'BASE TABLE' ORDER BY 1")}
    for t, c, kind, nullable in q("SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns"
                                  " WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"):
        if t in have:
            have[t].append((c, kind, nullable))
    indexes = {name: table for name, table in q("SELECT indexname, tablename FROM pg_indexes WHERE schemaname = 'public'")}
    views = [v for (v,) in q("SELECT table_name FROM information_schema.views WHERE table_schema = 'public'")]

    for t, cols in have.items():
        names = [c for c, _, _ in cols]
        rows = q(f'SELECT count(*) FROM "{t}"')[0][0]
        size = q("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))", (f'public."{t}"',))[0][0]
        out(f"\n{t}: {rows} rows, {size}{'' if t in want['tables'] else '   (not in schema.sql)'}")
        out("  columns: " + ", ".join(f"{c} {k}{'' if n == 'YES' else ' not null'}" for c, k, n in cols))
        out("  indexes: " + (", ".join(sorted(i for i, table in indexes.items() if table == t)) or "none"))
        if "circle_id" in names and rows:
            per = q(f'SELECT circle_id, count(*) FROM "{t}" GROUP BY 1 ORDER BY 2 DESC LIMIT 20')
            out("  per circle: " + ", ".join(f"{short(c)}={n}" for c, n in per))
        if "kind" in names and rows:
            out("  per kind: " + ", ".join(f"{k}={n}" for k, n in q(f'SELECT kind, count(*) FROM "{t}" GROUP BY 1 ORDER BY 1')))
        for c in TIME_COLUMNS:
            if c in names and rows:
                lo, hi = q(f'SELECT min("{c}"), max("{c}") FROM "{t}"')[0]
                out(f"  {c}: {str(lo)[:19]} .. {str(hi)[:19]}")
        if t == "presence" and rows:
            out(f"  unexpired now: {q('SELECT count(*) FROM presence WHERE expires_at > now()')[0][0]}")
    out("\nviews: " + (", ".join(sorted(views)) or "none"))

    wrong = []
    for t, cols in want["tables"].items():
        if t not in have:
            wrong.append(f"table {t} is in schema.sql but not in the database (apply schema.sql)")
            continue
        got = [c for c, _, _ in have[t]]
        wrong += [f"{t}.{c} is in schema.sql but not in the database (needs a migration: CREATE TABLE IF NOT EXISTS"
                  " does not add columns)" for c in cols if c not in got]
        wrong += [f"{t}.{c} is in the database but not in schema.sql" for c in got if c not in cols]
    wrong += [f"index {i} on {t} is in schema.sql but not in the database (apply schema.sql)"
              for i, t in want["indexes"].items() if i not in indexes]
    wrong += [f"view {v} is in schema.sql but not in the database (apply schema.sql)" for v in want["views"] if v not in views]
    wrong += [f"table {t} is in the database but not in schema.sql" for t in have if t not in want["tables"]]
    say("against schema.sql: " + ("matches" if not wrong else f"{len(wrong)} difference(s)"))
    for line in wrong:
        say("  - " + line)
    return wrong


CREATES = re.compile(r"CREATE (TABLE|INDEX) IF NOT EXISTS (\w+)|CREATE OR REPLACE (VIEW) (\w+)")


def statements(sql: str) -> list[tuple[str, str, str]]:
    """schema.sql as (kind, name, statement). Raises ValueError for anything that is not one of its three creates."""
    out = []
    for raw in re.sub(r"--[^\n]*", "", sql).split(";"):
        text = raw.strip()
        if not text:
            continue
        m = CREATES.match(text)
        if m is None:
            raise ValueError(f"schema.sql holds a statement --apply will not run: {text.split(chr(10))[0][:60]}")
        out.append(((m.group(1) or m.group(3)).lower(), m.group(2) or m.group(4), text))
    return out


def apply(conn, out=print) -> list[str]:
    """Create what schema.sql has and the database lacks, in one transaction. Never alters, replaces or drops:
    a table or index that exists is skipped by IF NOT EXISTS, and a view that exists is left exactly as it is."""
    todo = statements(SCHEMA.read_text())
    cur = conn.cursor()
    cur.execute("SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"
                " WHERE n.nspname = current_schema()")
    there = {name for (name,) in cur.fetchall()}
    created = []
    for kind, name, text in todo:
        if name in there:
            continue
        cur.execute(text)
        created.append(f"{kind} {name}")
    conn.commit()
    out("created: " + (", ".join(created) if created else "nothing (every table, index and view was already there)"))
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contexts.circle.schema", description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only: what differs from schema.sql (the default)")
    mode.add_argument("--counts", action="store_true", help="read-only: --check plus shapes and row counts")
    mode.add_argument("--apply", action="store_true", help="create what is missing; never alters or drops")
    args = parser.parse_args(argv)
    url = url_from_env()
    if not url:
        print(f"{ENV_URL} is not set (environment or .env): nothing to check")
        return 2
    transport = NeonTransport()  # this run's own: the tool decides afresh every time
    try:
        conn = transport.connect(url)
    except CircleOffline as e:  # the driver is missing and tcp was asked for, or the setting is not one of the three
        print(e)
        return 2
    except Exception as e:  # the driver's message names the host: say only what kind of failure it was
        print(f"could not connect ({type(e).__name__}); is port 5432 reachable from this network?"
              " ATLAS_NEON_TRANSPORT=https goes over port 443")
        return 1
    print("transport: " + {"tcp": "tcp (psycopg, port 5432)", "https": "https (port 443)"}[transport.chosen])
    try:
        if args.apply:
            apply(conn)
            conn.rollback()
        conn.read_only = True
        return 1 if audit(conn, counts=args.counts) else 0
    except NeonHttpsDropped as e:  # over https the first statement is the first contact; its message is ours, no host
        print(f"could not connect over https, port 443 ({e})")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
