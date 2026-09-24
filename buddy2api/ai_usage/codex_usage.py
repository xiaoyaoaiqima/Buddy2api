"""Read-only statistics for local Codex tasks and token events."""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_CODEX_STATE_DB = Path.home() / ".codex" / "state_5.sqlite"
DEFAULT_USAGE_CACHE_DB = Path(__file__).resolve().parents[2] / "data" / "codex_usage_cache.sqlite"
ALLOWED_RANGES = {"7", "30", "90", "all"}
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
PARSER_VERSION = 6
PROVIDER_ALIASES = {
    "goumanbuka": "custom",
    "raap": "cliproxy",
}
_CACHE_LOCK = threading.Lock()
_LAST_REFRESH: dict[str, float] = {}


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _range_start(range_key: str, now: datetime) -> datetime | None:
    if range_key == "all":
        return None
    return now - timedelta(days=int(range_key) - 1)


def _project_label(cwd: str) -> str:
    path = Path(cwd or "")
    return path.name or str(path) or "未知项目"


def _provider_label(value: Any) -> str:
    provider = str(value or "").strip()
    if not provider:
        return "未知 Provider"
    normalized = provider.casefold()
    if normalized == "openai":
        return "OpenAI"
    if normalized in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[normalized]
    return provider


def _compact_thread(row: sqlite3.Row) -> dict[str, Any]:
    created = datetime.fromtimestamp(int(row["created_at"]))
    updated = datetime.fromtimestamp(int(row["updated_at"]))
    return {
        "id": row["id"],
        "title": row["title"] or row["first_user_message"] or "未命名任务",
        "project": _project_label(row["cwd"]),
        "cwd": row["cwd"],
        "model": row["model"] or "未知模型",
        "provider": _provider_label(row["model_provider"]),
        "source": row["source"],
        "tokens": int(row["tokens_used"] or 0),
        "created_at": created.isoformat(timespec="seconds"),
        "updated_at": updated.isoformat(timespec="seconds"),
        "archived": bool(row["archived"]),
    }


def _cache_path_for(state_db: Path, cache_db: str | Path | None) -> Path:
    if cache_db is not None:
        return Path(cache_db).expanduser()
    if state_db.resolve() == DEFAULT_CODEX_STATE_DB.resolve():
        return DEFAULT_USAGE_CACHE_DB
    return state_db.with_name(f"{state_db.stem}.codex_usage_cache.sqlite")


def _init_cache(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS rollout_index (
            thread_id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            source TEXT NOT NULL,
            model_provider TEXT NOT NULL DEFAULT '',
            is_subagent INTEGER NOT NULL,
            parser_version INTEGER NOT NULL DEFAULT 1,
            parsed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_usage (
            thread_id TEXT NOT NULL,
            usage_date TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            cached_input_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (thread_id, usage_date)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_usage_date ON daily_usage(usage_date);
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(rollout_index)")}
    if "parser_version" not in columns:
        conn.execute(
            "ALTER TABLE rollout_index ADD COLUMN parser_version INTEGER NOT NULL DEFAULT 1"
        )
    if "model_provider" not in columns:
        conn.execute(
            "ALTER TABLE rollout_index ADD COLUMN model_provider TEXT NOT NULL DEFAULT ''"
        )
    if "shard_fingerprint" not in columns:
        conn.execute("ALTER TABLE rollout_index ADD COLUMN shard_fingerprint TEXT NOT NULL DEFAULT ''")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(LOCAL_TZ)


def _parent_thread_id(payload: dict[str, Any]) -> str | None:
    parent_id = payload.get("forked_from_id") or payload.get("parent_thread_id")
    if isinstance(parent_id, str) and parent_id:
        return parent_id
    source = payload.get("source") or {}
    if not isinstance(source, dict):
        return None
    thread_spawn = ((source.get("subagent") or {}).get("thread_spawn") or {})
    parent_id = thread_spawn.get("parent_thread_id")
    return parent_id if isinstance(parent_id, str) and parent_id else None


def _read_rollout(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    """Read token events and the original fork parent from one rollout file."""
    parent_id: str | None = None
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if item.get("type") == "session_meta" and parent_id is None:
                parent_id = _parent_thread_id(item.get("payload") or {})
            payload = item.get("payload") or {}
            if item.get("type") != "event_msg" or payload.get("type") != "token_count":
                continue
            info = payload.get("info") or {}
            totals = info.get("total_token_usage") or {}
            event_time = _parse_timestamp(item.get("timestamp"))
            if not totals or event_time is None:
                continue
            events.append(
                {
                    "info": info,
                    "totals": totals,
                    "last": info.get("last_token_usage") or {},
                    "event_time": event_time,
                }
            )
    return parent_id, events


def _inherited_prefix_length(
    events: list[dict[str, Any]], parent_events: list[dict[str, Any]]
) -> int:
    """Find the copied parent event prefix present in a forked rollout.

    Older rollouts omit zero-valued token fields that newer writers emit
    explicitly.  Compare normalized cumulative counters instead of the raw
    event object, otherwise the same inherited history can appear different.
    """
    prefix = 0
    limit = min(len(events), len(parent_events))
    while (
        prefix < limit
        and _normalized_token_usage(events[prefix]["totals"])
        == _normalized_token_usage(parent_events[prefix]["totals"])
    ):
        prefix += 1
    return prefix


def _normalized_token_usage(values: dict[str, Any]) -> tuple[int, ...]:
    """Return all supported token counters with omitted fields treated as zero."""
    return tuple(max(0, int(values.get(field) or 0)) for field in TOKEN_FIELDS)


def _discover_rollout_shards(
    threads: list[sqlite3.Row], state_db: Path
) -> dict[str, list[Path]]:
    """The state DB points at the latest shard, not necessarily the full history."""
    paths = {str(t["id"]): {Path(t["rollout_path"]).expanduser()} for t in threads}
    homes = {state_db.parent}
    for group in paths.values():
        for path in group:
            for ancestor in path.parents:
                if ancestor.name in {"sessions", "archived_sessions"}:
                    homes.add(ancestor.parent)
                    break
    pattern = re.compile(r"-([0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})(?:_|\.jsonl$)")
    for home in homes:
        for directory in (home / "sessions", home / "archived_sessions"):
            if not directory.is_dir():
                continue
            for path in directory.rglob("rollout-*.jsonl"):
                match = pattern.search(path.name)
                if match and match[1] in paths:
                    paths[match[1]].add(path)
    return {key: sorted(group) for key, group in paths.items()}


def _shard_fingerprint(paths: list[Path]) -> str:
    stamps = []
    for path in paths:
        try:
            stat = path.stat()
            stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            stamps.append((str(path), None, None))
    return hashlib.sha256(json.dumps(stamps).encode()).hexdigest()


def _read_rollout_shards(paths: list[Path]) -> tuple[str | None, list[dict[str, Any]]]:
    parent_id = None
    events = []
    seen = set()
    first_path = None
    first_time = None
    for path in paths:
        if not path.exists():
            continue
        shard_parent, shard_events = _read_rollout(path)
        parent_id = parent_id or shard_parent
        if shard_events:
            shard_time = min(event["event_time"] for event in shard_events)
            if (first_time is None or shard_time < first_time
                    or (shard_time == first_time and "_" not in path.stem)):
                first_time, first_path = shard_time, path
        for event in shard_events:
            key = (event["event_time"], _normalized_token_usage(event["totals"]))
            if key not in seen:
                events.append(event)
                seen.add(key)
    events.sort(key=lambda event: event["event_time"])
    # If only a rotated fragment remains, its carried cumulative counter has
    # no known date. Count its first request only, never assign the carry to today.
    if events and first_path and "_" in first_path.stem and events[0]["last"]:
        events[0] = {**events[0], "carried_baseline": True}
    return parent_id, events


def _summarize_rollout_events(
    events: list[dict[str, Any]], *, inherited_events: list[dict[str, Any]] | None = None
) -> dict[str, dict[str, int]]:
    """Convert cumulative token_count events into local-date positive deltas.

    Forked sub-agent rollouts may contain a byte-for-byte copy of their parent
    history. Older rollout formats can instead begin with the parent's
    cumulative counter but omit the copied events. In both cases, only the
    child-specific increments are counted.
    """
    daily: dict[str, dict[str, int]] = defaultdict(
        lambda: {field: 0 for field in TOKEN_FIELDS}
    )
    high_water = {field: 0 for field in TOKEN_FIELDS}
    start_index = 0
    if events and events[0].get("carried_baseline"):
        high_water = {
            field: max(0, int(events[0]["totals"].get(field) or 0)
                       - int(events[0]["last"].get(field) or 0))
            for field in TOKEN_FIELDS
        }
    if inherited_events and events:
        start_index = _inherited_prefix_length(events, inherited_events)
        if start_index:
            baseline = events[start_index - 1]["totals"]
            high_water = {
                field: max(0, int(baseline.get(field) or 0)) for field in TOKEN_FIELDS
            }
        elif events[0]["last"]:
            # Some legacy forks begin at the inherited cumulative total without
            # copying the individual parent events. The first request delta is
            # the only reliable child contribution before the first event.
            totals = events[0]["totals"]
            last = events[0]["last"]
            high_water = {
                field: max(0, int(totals.get(field) or 0) - int(last.get(field) or 0))
                for field in TOKEN_FIELDS
            }

    for event in events[start_index:]:
        totals = event["totals"]
        total_now = max(0, int(totals.get("total_tokens") or 0))
        total_high = high_water["total_tokens"]
        is_reset = total_high > 0 and total_now < total_high * 0.5
        deltas: dict[str, int] = {}
        for field in TOKEN_FIELDS:
            current = max(0, int(totals.get(field) or 0))
            if is_reset:
                deltas[field] = current
                high_water[field] = current
            elif current >= high_water[field]:
                deltas[field] = current - high_water[field]
                high_water[field] = current
            else:
                # Token events can arrive slightly out of order. Keep the
                # high-water mark so a small dip is not counted twice.
                deltas[field] = 0
        if deltas["total_tokens"] > 0:
            day = event["event_time"].date().isoformat()
            for field, delta in deltas.items():
                daily[day][field] += delta
    return {
        usage_date: values
        for usage_date, values in daily.items()
        if values["total_tokens"] > 0
    }


def _parse_rollout(
    path: Path, *, inherited_events: list[dict[str, Any]] | None = None
) -> dict[str, dict[str, int]]:
    """Read and summarize one rollout file for callers outside cache refresh."""
    _, events = _read_rollout(path)
    return _summarize_rollout_events(events, inherited_events=inherited_events)


def _refresh_usage_cache(
    *,
    state_db: Path,
    cache_db: Path,
    ttl_seconds: float,
) -> None:
    cache_key = str(cache_db.resolve())
    with _CACHE_LOCK:
        if ttl_seconds > 0 and time.monotonic() - _LAST_REFRESH.get(cache_key, 0) < ttl_seconds:
            return
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        with _connect_read_only(state_db) as state_conn:
            thread_columns = {
                str(row["name"])
                for row in state_conn.execute("PRAGMA table_info(threads)").fetchall()
            }
            provider_column = (
                "model_provider"
                if "model_provider" in thread_columns
                else "'' AS model_provider"
            )
            threads = state_conn.execute(
                f"""
                SELECT id, rollout_path, source, {provider_column}
                FROM threads
                WHERE rollout_path IS NOT NULL AND rollout_path != ''
                """
            ).fetchall()

        threads_by_id = {str(thread["id"]): thread for thread in threads}
        shard_paths = _discover_rollout_shards(threads, state_db)
        fingerprints = {key: _shard_fingerprint(paths) for key, paths in shard_paths.items()}
        rollout_events: dict[str, list[dict[str, Any]]] = {}
        rollout_parents: dict[str, str | None] = {}

        def load_rollout(thread_id: str) -> tuple[str | None, list[dict[str, Any]]]:
            if thread_id in rollout_events:
                return rollout_parents[thread_id], rollout_events[thread_id]
            thread = threads_by_id.get(thread_id)
            if thread is None:
                return None, []
            try:
                parent_id, events = _read_rollout_shards(shard_paths[thread_id])
            except OSError:
                return None, []
            rollout_parents[thread_id] = parent_id
            rollout_events[thread_id] = events
            return parent_id, events

        with sqlite3.connect(cache_db, timeout=30) as cache_conn:
            cache_conn.row_factory = sqlite3.Row
            _init_cache(cache_conn)
            indexed = {
                row["thread_id"]: row
                for row in cache_conn.execute("SELECT * FROM rollout_index").fetchall()
            }
            live_ids: set[str] = set()
            for thread in threads:
                thread_id = str(thread["id"])
                live_ids.add(thread_id)
                rollout_path = Path(str(thread["rollout_path"])).expanduser()
                try:
                    stat = rollout_path.stat()
                except OSError:
                    continue
                prior = indexed.get(thread_id)
                provider_name = _provider_label(thread["model_provider"])
                if (
                    prior is not None
                    and prior["rollout_path"] == str(rollout_path)
                    and int(prior["size"]) == stat.st_size
                    and int(prior["mtime_ns"]) == stat.st_mtime_ns
                    and int(prior["parser_version"]) == PARSER_VERSION
                    and str(prior["model_provider"]) == provider_name
                    and str(prior["shard_fingerprint"]) == fingerprints[thread_id]
                ):
                    continue
                try:
                    parent_id, events = load_rollout(thread_id)
                    _, parent_events = load_rollout(parent_id) if parent_id else (None, [])
                    daily = _summarize_rollout_events(
                        events, inherited_events=parent_events
                    )
                except OSError:
                    continue
                cache_conn.execute("DELETE FROM daily_usage WHERE thread_id = ?", (thread_id,))
                cache_conn.executemany(
                    """
                    INSERT INTO daily_usage (
                        thread_id, usage_date, input_tokens, cached_input_tokens,
                        cache_write_input_tokens, output_tokens,
                        reasoning_output_tokens, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (thread_id, usage_date, *(values[field] for field in TOKEN_FIELDS))
                        for usage_date, values in daily.items()
                    ],
                )
                source = str(thread["source"] or "")
                cache_conn.execute(
                    """
                    INSERT OR REPLACE INTO rollout_index
                    (thread_id, rollout_path, size, mtime_ns, source, model_provider,
                     is_subagent, parser_version, parsed_at, shard_fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        str(rollout_path),
                        stat.st_size,
                        stat.st_mtime_ns,
                        source,
                        provider_name,
                        int(source.startswith('{"subagent"')),
                        PARSER_VERSION,
                        datetime.now().isoformat(timespec="seconds"),
                        fingerprints[thread_id],
                    ),
                )
            stale_ids = set(indexed) - live_ids
            if stale_ids:
                cache_conn.executemany(
                    "DELETE FROM daily_usage WHERE thread_id = ?",
                    [(thread_id,) for thread_id in stale_ids],
                )
                cache_conn.executemany(
                    "DELETE FROM rollout_index WHERE thread_id = ?",
                    [(thread_id,) for thread_id in stale_ids],
                )
            cache_conn.commit()
        _LAST_REFRESH[cache_key] = time.monotonic()


def _load_daily_usage(
    cache_db: Path,
    start_date: str | None,
    end_date: str,
    provider: str | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    where = "WHERE d.usage_date <= ?"
    params: list[Any] = [end_date]
    if start_date is not None:
        where += " AND d.usage_date >= ?"
        params.append(start_date)
    if provider is not None:
        where += " AND i.model_provider = ?"
        params.append(provider)
    with sqlite3.connect(cache_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT d.usage_date,
                   SUM(d.input_tokens) AS input_tokens,
                   SUM(d.cached_input_tokens) AS cached_input_tokens,
                   SUM(d.cache_write_input_tokens) AS cache_write_input_tokens,
                   SUM(d.output_tokens) AS output_tokens,
                   SUM(d.reasoning_output_tokens) AS reasoning_output_tokens,
                   SUM(d.total_tokens) AS total_tokens,
                   SUM(CASE WHEN i.is_subagent = 0 THEN d.total_tokens ELSE 0 END) AS root_tokens,
                   SUM(CASE WHEN i.is_subagent = 1 THEN d.total_tokens ELSE 0 END) AS subagent_tokens,
                   COUNT(DISTINCT CASE WHEN d.total_tokens > 0 THEN d.thread_id END) AS involved_threads,
                   COUNT(DISTINCT CASE WHEN d.total_tokens > 0 AND i.is_subagent = 0 THEN d.thread_id END) AS root_threads,
                   COUNT(DISTINCT CASE WHEN d.total_tokens > 0 AND i.is_subagent = 1 THEN d.thread_id END) AS subagent_threads
            FROM daily_usage d
            JOIN rollout_index i ON i.thread_id = d.thread_id
            {where}
            GROUP BY d.usage_date
            ORDER BY d.usage_date
            """,
            params,
        ).fetchall()
    daily = {str(row["usage_date"]): dict(row) for row in rows}
    totals = {field: sum(int(row[field] or 0) for row in rows) for field in TOKEN_FIELDS}
    totals["root_tokens"] = sum(int(row["root_tokens"] or 0) for row in rows)
    totals["subagent_tokens"] = sum(int(row["subagent_tokens"] or 0) for row in rows)
    totals["involved_threads"] = len(
        {
            thread_id
            for thread_id in _usage_thread_ids(
                cache_db, start_date, end_date, provider=provider
            )
        }
    )
    return daily, totals


def _usage_thread_ids(
    cache_db: Path,
    start_date: str | None,
    end_date: str,
    provider: str | None = None,
) -> list[str]:
    where = "WHERE d.usage_date <= ? AND d.total_tokens > 0"
    params: list[Any] = [end_date]
    if start_date is not None:
        where += " AND d.usage_date >= ?"
        params.append(start_date)
    if provider is not None:
        where += " AND i.model_provider = ?"
        params.append(provider)
    with sqlite3.connect(cache_db) as conn:
        return [
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT DISTINCT d.thread_id
                FROM daily_usage d
                JOIN rollout_index i ON i.thread_id = d.thread_id
                {where}
                """,
                params,
            ).fetchall()
        ]


def _usage_by_thread(
    cache_db: Path, start_date: str | None, end_date: str
) -> dict[str, int]:
    where = "WHERE usage_date <= ?"
    params: list[Any] = [end_date]
    if start_date is not None:
        where += " AND usage_date >= ?"
        params.append(start_date)
    with sqlite3.connect(cache_db) as conn:
        conn.row_factory = sqlite3.Row
        return {
            str(row["thread_id"]): int(row["tokens"] or 0)
            for row in conn.execute(
                f"""
                SELECT thread_id, SUM(total_tokens) AS tokens
                FROM daily_usage
                {where}
                GROUP BY thread_id
                """,
                params,
            ).fetchall()
        }


def _load_project_daily_usage(
    cache_db: Path,
    start_date: str | None,
    end_date: str,
    cwd_by_thread: dict[str, str],
    provider: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group positive daily Token activity by the task's working directory."""
    where = "WHERE d.usage_date <= ? AND d.total_tokens > 0"
    params: list[Any] = [end_date]
    if start_date is not None:
        where += " AND d.usage_date >= ?"
        params.append(start_date)
    if provider is not None:
        where += " AND i.model_provider = ?"
        params.append(provider)
    with sqlite3.connect(cache_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT d.usage_date, d.thread_id, d.total_tokens
            FROM daily_usage d
            JOIN rollout_index i ON i.thread_id = d.thread_id
            {where}
            """,
            params,
        ).fetchall()

    projects_by_date: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cwd = cwd_by_thread.get(str(row["thread_id"]), "")
        project = projects_by_date[str(row["usage_date"])].setdefault(
            cwd,
            {"name": _project_label(cwd), "cwd": cwd, "tasks": 0, "tokens": 0},
        )
        project["tasks"] += 1
        project["tokens"] += int(row["total_tokens"] or 0)
    return {
        date: sorted(values.values(), key=lambda item: (-item["tokens"], -item["tasks"]))
        for date, values in projects_by_date.items()
    }


def build_codex_usage_stats(
    *,
    state_db: str | Path = DEFAULT_CODEX_STATE_DB,
    cache_db: str | Path | None = None,
    range_key: str = "30",
    provider: str = "all",
    now: datetime | None = None,
    cache_ttl_seconds: float = 5,
) -> dict[str, Any]:
    """Aggregate task creation and actual event-time token deltas."""
    if range_key not in ALLOWED_RANGES:
        raise ValueError(f"unsupported range: {range_key}")
    provider_value = str(provider or "all").strip()
    selected_provider = (
        None if not provider_value or provider_value.casefold() == "all" else _provider_label(provider_value)
    )
    db_path = Path(state_db).expanduser()
    if not db_path.is_file():
        raise FileNotFoundError(f"Codex state database not found: {db_path}")

    now = now or datetime.now()
    start = _range_start(range_key, now)
    start_date = start.date().isoformat() if start is not None else None
    end_date = now.date().isoformat()
    usage_cache = _cache_path_for(db_path, cache_db)
    _refresh_usage_cache(
        state_db=db_path, cache_db=usage_cache, ttl_seconds=cache_ttl_seconds
    )

    params: list[Any] = []
    where = "WHERE source NOT LIKE '{\"subagent\"%'"
    if start is not None:
        where += " AND created_at >= ?"
        params.append(int(start.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()))

    with _connect_read_only(db_path) as conn:
        thread_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(threads)").fetchall()
        }
        provider_column = (
            "model_provider" if "model_provider" in thread_columns else "'' AS model_provider"
        )
        all_rows = conn.execute(
            f"""
            SELECT id, title, first_user_message, cwd, model, {provider_column}, source,
                   tokens_used, created_at, updated_at, archived
            FROM threads
            {where}
            ORDER BY created_at ASC
            """,
            params,
        ).fetchall()
        subagent_rows = conn.execute(
            f"""
            SELECT {provider_column}, tokens_used
            FROM threads
            WHERE source LIKE '{{"subagent"%'
            {"AND created_at >= ?" if start is not None else ""}
            """,
            params,
        ).fetchall()
        all_thread_rows = conn.execute(
            f"SELECT id, cwd FROM threads"
        ).fetchall()

    usage_by_thread = _usage_by_thread(usage_cache, start_date, end_date)
    cwd_by_thread = {
        str(row["id"]): str(row["cwd"] or "") for row in all_thread_rows
    }
    providers: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        provider_name = _provider_label(row["model_provider"])
        provider_row = providers.setdefault(
            provider_name, {"name": provider_name, "tasks": 0, "tokens": 0}
        )
        provider_row["tasks"] += 1
        provider_row["tokens"] += usage_by_thread.get(str(row["id"]), 0)
    provider_rows = sorted(
        providers.values(), key=lambda item: (-item["tasks"], -item["tokens"])
    )
    rows = [
        row
        for row in all_rows
        if selected_provider is None
        or _provider_label(row["model_provider"]) == selected_provider
    ]
    filtered_subagent_rows = [
        row
        for row in subagent_rows
        if selected_provider is None
        or _provider_label(row["model_provider"]) == selected_provider
    ]
    usage_daily, usage_totals = _load_daily_usage(
        usage_cache, start_date, end_date, provider=selected_provider
    )
    project_daily = _load_project_daily_usage(
        usage_cache,
        start_date,
        end_date,
        cwd_by_thread,
        provider=selected_provider,
    )
    created_daily: dict[str, int] = {}
    projects: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    sources: dict[str, int] = {}
    archived = 0
    for row in rows:
        date_key = datetime.fromtimestamp(int(row["created_at"])).strftime("%Y-%m-%d")
        created_daily[date_key] = created_daily.get(date_key, 0) + 1
        # Project/model rankings must use the same event-time, de-duplicated
        # token window as the chart and Provider filter, rather than the
        # thread's lifetime index counter.
        tokens = int(usage_by_thread.get(str(row["id"]), 0))
        archived += int(bool(row["archived"]))
        cwd = row["cwd"] or ""
        project = projects.setdefault(
            cwd, {"name": _project_label(cwd), "cwd": cwd, "tasks": 0, "tokens": 0}
        )
        project["tasks"] += 1
        project["tokens"] += tokens
        model_name = row["model"] or "未知模型"
        model = models.setdefault(model_name, {"name": model_name, "tasks": 0, "tokens": 0})
        model["tasks"] += 1
        model["tokens"] += tokens
        sources[str(row["source"])] = sources.get(str(row["source"]), 0) + 1

    all_dates = set(created_daily) | set(usage_daily)
    if range_key == "all":
        first_day = min(
            (datetime.strptime(date, "%Y-%m-%d").date() for date in all_dates),
            default=None,
        )
    else:
        first_day = start.date() if start is not None else None
    daily_rows: list[dict[str, Any]] = []
    if first_day is not None:
        cursor = first_day
        while cursor <= now.date():
            date_key = cursor.isoformat()
            usage = usage_daily.get(date_key, {})
            input_tokens = int(usage.get("input_tokens") or 0)
            cached_tokens = int(usage.get("cached_input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)
            involved_threads = int(usage.get("involved_threads") or 0)
            daily_rows.append(
                {
                    "date": date_key,
                    "tasks": created_daily.get(date_key, 0),
                    "tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "non_cached_input_tokens": max(0, input_tokens - cached_tokens),
                    "output_tokens": output_tokens,
                    "unclassified_tokens": max(0, total_tokens - input_tokens - output_tokens),
                    "involved_threads": involved_threads,
                    "root_threads": int(usage.get("root_threads") or 0),
                    "subagent_threads": int(usage.get("subagent_threads") or 0),
                    "root_tokens": int(usage.get("root_tokens") or 0),
                    "subagent_tokens": int(usage.get("subagent_tokens") or 0),
                    "tokens_per_thread": round(total_tokens / involved_threads)
                    if involved_threads
                    else 0,
                    "cache_ratio": round(cached_tokens / input_tokens, 4)
                    if input_tokens
                    else 0,
                }
            )
            cursor += timedelta(days=1)

    project_rows = sorted(projects.values(), key=lambda item: (-item["tasks"], -item["tokens"]))
    model_rows = sorted(models.values(), key=lambda item: (-item["tasks"], -item["tokens"]))
    total_input = int(usage_totals["input_tokens"])
    total_cached = int(usage_totals["cached_input_tokens"])
    total_tokens = int(usage_totals["total_tokens"])
    token_active_dates = [date for date, usage in usage_daily.items() if int(usage.get("total_tokens") or 0) > 0]
    first_date = min(all_dates) if all_dates else None
    last_date = max(all_dates) if all_dates else None

    return {
        "range": range_key,
        "range_label": "全部" if range_key == "all" else f"近 {range_key} 日",
        "provider": selected_provider or "all",
        "provider_label": selected_provider or "全部 Provider",
        "data_source": str(db_path),
        "usage_cache": str(usage_cache),
        "generated_at": now.isoformat(timespec="seconds"),
        "methodology": "任务数按创建日；Token 按 rollout token_count 事件的累计值正增量归入事件发生日（北京时间）。",
        "summary": {
            "tasks": len(rows),
            "tokens": total_tokens,
            "input_tokens": total_input,
            "cached_input_tokens": total_cached,
            "non_cached_input_tokens": max(0, total_input - total_cached),
            "output_tokens": int(usage_totals["output_tokens"]),
            "cache_ratio": round(total_cached / total_input, 4) if total_input else 0,
            "active_days": len(token_active_dates),
            "projects": len(project_rows),
            "archived_tasks": archived,
            "avg_tokens_per_task": round(total_tokens / usage_totals["involved_threads"])
            if usage_totals["involved_threads"]
            else 0,
            "involved_threads": int(usage_totals["involved_threads"]),
            "subagent_tasks": len(filtered_subagent_rows),
            "root_tokens": int(usage_totals["root_tokens"]),
            "subagent_tokens": int(usage_totals["subagent_tokens"]),
            "first_date": first_date,
            "last_date": last_date,
        },
        "daily": daily_rows,
        "projects": project_rows[:12],
        "project_daily": project_daily,
        "models": model_rows,
        "providers": provider_rows,
        "sources": [
            {"name": name, "tasks": count}
            for name, count in sorted(sources.items(), key=lambda item: -item[1])
        ],
        "recent_tasks": [_compact_thread(row) for row in reversed(rows[-20:])],
    }
