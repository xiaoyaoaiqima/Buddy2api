"""Unified, read-only local usage statistics for AI coding tools."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from buddy2api.ai_usage.codex_usage import (
    ALLOWED_RANGES,
    LOCAL_TZ,
    _project_label,
    _provider_label,
    _range_start,
    build_codex_usage_stats,
)


TOOL_LABELS = {
    "codex": "Codex",
    "claude": "Claude Code",
    "zcode": "ZCode",
    "workbuddy": "WorkBuddy",
    # 网关自己转发的流量。其它四个读的是客户端留下的轨迹，这个读的是经过网关的请求，
    # 两者互补：客户端轨迹有项目/会话结构但看不到直连 API 的调用，网关日志反之。
    "gateway": "网关直连",
}
ALLOWED_TOOLS = {"all", *TOOL_LABELS}
DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"
DEFAULT_ZCODE_DB = Path.home() / ".zcode" / "cli" / "db" / "db.sqlite"
DEFAULT_WORKBUDDY_DB = Path.home() / ".workbuddy" / "workbuddy.db"
DEFAULT_WORKBUDDY_TRACES = Path.home() / ".workbuddy" / "traces"
DEFAULT_AI_USAGE_CACHE_DB = Path(__file__).resolve().parents[2] / "data" / "ai_usage_cache.sqlite"
# 网关自己的库。与 database.DB_PATH 保持同一来源，避免两处各写各的。
# 用函数而不是常量：测试会 monkeypatch CB_GATEWAY_DB_PATH，模块级常量会读成旧值。
def default_codex_state_db() -> Path:
    from buddy2api.ai_usage.codex_usage import DEFAULT_CODEX_STATE_DB

    return Path(DEFAULT_CODEX_STATE_DB)


def default_gateway_db() -> Path:
    import os

    from buddy2api.paths import PROJECT_ROOT

    return Path(os.environ.get("CB_GATEWAY_DB_PATH", PROJECT_ROOT / "codebuddy_gateway.db"))
EXTERNAL_PARSER_VERSION = 1
HOT_WINDOW_DAYS = 2

_CACHE_LOCK = threading.Lock()
_SOURCE_CACHE: dict[str, tuple[float, tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {}


def _init_external_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_cache (
            tool TEXT NOT NULL,
            source_group TEXT NOT NULL,
            source_key TEXT NOT NULL,
            size INTEGER NOT NULL DEFAULT 0,
            mtime_ns INTEGER NOT NULL DEFAULT 0,
            source_time INTEGER NOT NULL DEFAULT 0,
            parser_version INTEGER NOT NULL,
            sessions_json TEXT NOT NULL DEFAULT '[]',
            events_json TEXT NOT NULL DEFAULT '[]',
            parsed_at TEXT NOT NULL,
            PRIMARY KEY (tool, source_group, source_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_cache_group "
        "ON source_cache(tool, source_group, source_time)"
    )


def _records_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value),
    )


def _records_from_json(value: str, *, sessions: bool = False) -> list[dict[str, Any]]:
    try:
        records = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(records, list):
        return []
    if sessions:
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in ("created_at", "updated_at"):
                parsed = _parse_time(record.get(field))
                if parsed:
                    record[field] = parsed
    return [record for record in records if isinstance(record, dict)]


def _cache_group_rows(
    conn: sqlite3.Connection, tool: str, source_group: str
) -> dict[str, sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return {
        str(row["source_key"]): row
        for row in conn.execute(
            "SELECT * FROM source_cache WHERE tool = ? AND source_group = ?",
            (tool, source_group),
        ).fetchall()
    }


def _upsert_cache_item(
    conn: sqlite3.Connection,
    *,
    tool: str,
    source_group: str,
    source_key: str,
    sessions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    size: int = 0,
    mtime_ns: int = 0,
    source_time: int = 0,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO source_cache (
            tool, source_group, source_key, size, mtime_ns, source_time,
            parser_version, sessions_json, events_json, parsed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tool,
            source_group,
            source_key,
            size,
            mtime_ns,
            source_time,
            EXTERNAL_PARSER_VERSION,
            _records_json(sessions),
            _records_json(events),
            datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        ),
    )


def _cached_source(
    key: str,
    loader: Callable[[], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    ttl_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with _CACHE_LOCK:
        cached = _SOURCE_CACHE.get(key)
        if cached and time.monotonic() - cached[0] < ttl_seconds:
            return cached[1]
    value = loader()
    with _CACHE_LOCK:
        _SOURCE_CACHE[key] = (time.monotonic(), value)
    return value


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(LOCAL_TZ)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(LOCAL_TZ)


def _provider_from_model(model: str, fallback: str) -> str:
    normalized = str(model or "").casefold()
    if normalized.startswith(("claude", "hy")):
        return "Anthropic"
    if normalized.startswith(("gpt", "o1", "o3", "o4")):
        return "OpenAI"
    if "deepseek" in normalized:
        return "DeepSeek"
    if normalized.startswith("glm"):
        return "Zhipu"
    if normalized.startswith(("doubao", "seed")):
        return "Volcengine"
    return fallback


def _session(
    *,
    session_id: str,
    tool: str,
    cwd: str,
    title: str,
    model: str,
    provider: str,
    created_at: datetime,
    updated_at: datetime,
    is_subagent: bool,
    archived: bool = False,
) -> dict[str, Any]:
    return {
        "id": session_id,
        "tool": tool,
        "cwd": cwd,
        "title": title or f"{TOOL_LABELS[tool]} 会话",
        "model": model or "未知模型",
        "provider": provider,
        "created_at": created_at,
        "updated_at": updated_at,
        "is_subagent": is_subagent,
        "archived": archived,
    }


def _event(
    *,
    session_id: str,
    tool: str,
    event_time: datetime,
    cwd: str,
    model: str,
    provider: str,
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    is_subagent: bool,
    requests: int = 1,
    tool_calls: int = 0,
    errors: int = 0,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "tool": tool,
        "date": event_time.date().isoformat(),
        "cwd": cwd,
        "model": model or "未知模型",
        "provider": provider,
        "input_tokens": max(0, int(input_tokens or 0)),
        "cached_input_tokens": max(0, int(cached_input_tokens or 0)),
        "cache_write_input_tokens": max(0, int(cache_write_input_tokens or 0)),
        "output_tokens": max(0, int(output_tokens or 0)),
        "total_tokens": max(0, int(total_tokens or 0)),
        "is_subagent": is_subagent,
        "requests": max(0, int(requests or 0)),
        "tool_calls": max(0, int(tool_calls or 0)),
        "errors": max(0, int(errors or 0)),
    }


def _load_claude_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    is_subagent = path.name.startswith("agent-")
    session_id = path.stem
    cwd = ""
    title = ""
    model = ""
    provider = "Claude Code"
    first_time: datetime | None = None
    last_time: datetime | None = None
    seen_messages: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return sessions, events
    with handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            event_time = _parse_time(item.get("timestamp"))
            if event_time:
                first_time = min(first_time, event_time) if first_time else event_time
                last_time = max(last_time, event_time) if last_time else event_time
            cwd = str(item.get("cwd") or cwd)
            title = str(item.get("slug") or title)
            message = item.get("message") or {}
            if not isinstance(message, dict):
                continue
            current_model = str(message.get("model") or model)
            if current_model and current_model != "<synthetic>":
                model = current_model
                provider = _provider_from_model(model, "Claude Code")
            usage = message.get("usage") or {}
            if not usage or not event_time or current_model == "<synthetic>":
                continue
            message_id = str(message.get("id") or item.get("uuid") or "")
            if message_id and message_id in seen_messages:
                continue
            if message_id:
                seen_messages.add(message_id)
            raw_input = int(usage.get("input_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_write = int(usage.get("cache_creation_input_tokens") or 0)
            output = int(usage.get("output_tokens") or 0)
            total_input = raw_input + cache_read + cache_write
            content = message.get("content") or []
            tool_calls = (
                sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                )
                if isinstance(content, list)
                else 0
            )
            events.append(
                _event(
                    session_id=session_id,
                    tool="claude",
                    event_time=event_time,
                    cwd=cwd,
                    model=current_model,
                    provider=_provider_from_model(current_model, "Claude Code"),
                    input_tokens=total_input,
                    cached_input_tokens=cache_read,
                    cache_write_input_tokens=cache_write,
                    output_tokens=output,
                    total_tokens=total_input + output,
                    is_subagent=is_subagent,
                    tool_calls=tool_calls,
                    errors=int(bool(item.get("isApiErrorMessage"))),
                )
            )
    if first_time:
        sessions.append(
            _session(
                session_id=session_id,
                tool="claude",
                cwd=cwd,
                title=title,
                model=model,
                provider=provider,
                created_at=first_time,
                updated_at=last_time or first_time,
                is_subagent=is_subagent,
            )
        )
    return sessions, events


def _load_claude(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if not root.is_dir():
        return sessions, events
    for path in root.rglob("*.jsonl"):
        file_sessions, file_events = _load_claude_file(path)
        sessions.extend(file_sessions)
        events.extend(file_events)
    return sessions, events


def _load_claude_cached(
    root: Path, cache_db: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    source_group = str(root.resolve())
    paths = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_db, timeout=30) as conn:
        _init_external_cache(conn)
        indexed = _cache_group_rows(conn, "claude", source_group)
        live_keys: set[str] = set()
        for path in paths:
            source_key = str(path.resolve())
            live_keys.add(source_key)
            try:
                stat = path.stat()
            except OSError:
                continue
            prior = indexed.get(source_key)
            if (
                prior is not None
                and int(prior["size"]) == stat.st_size
                and int(prior["mtime_ns"]) == stat.st_mtime_ns
                and int(prior["parser_version"]) == EXTERNAL_PARSER_VERSION
            ):
                file_sessions = _records_from_json(prior["sessions_json"], sessions=True)
                file_events = _records_from_json(prior["events_json"])
            else:
                file_sessions, file_events = _load_claude_file(path)
                _upsert_cache_item(
                    conn,
                    tool="claude",
                    source_group=source_group,
                    source_key=source_key,
                    sessions=file_sessions,
                    events=file_events,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            sessions.extend(file_sessions)
            events.extend(file_events)
        stale_keys = set(indexed) - live_keys
        if stale_keys:
            conn.executemany(
                "DELETE FROM source_cache WHERE tool = 'claude' AND source_group = ? AND source_key = ?",
                [(source_group, source_key) for source_key in stale_keys],
            )
        conn.commit()
    return sessions, events


def _load_zcode(db_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if not db_path.is_file():
        return sessions, events
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        session_rows = conn.execute(
            "SELECT id, parent_id, directory, title, time_created, time_updated, time_archived FROM session"
        ).fetchall()
        session_map = {str(row["id"]): row for row in session_rows}
        session_models: dict[str, tuple[str, str]] = {}
        usage_rows = conn.execute(
            """
            SELECT session_id, provider_id, model_id, started_at, input_tokens,
                   output_tokens, reasoning_tokens, cache_creation_input_tokens,
                   cache_read_input_tokens, computed_total_tokens, tool_call_count,
                   status
            FROM model_usage
            WHERE status IN ('completed', 'error', 'cancelled')
            """
        ).fetchall()
        for row in usage_rows:
            session_row = session_map.get(str(row["session_id"]))
            if not session_row:
                continue
            event_time = _parse_time(row["started_at"])
            if not event_time:
                continue
            model = str(row["model_id"] or "未知模型")
            provider = _provider_from_model(model, "ZCode")
            session_models[str(row["session_id"])] = (model, provider)
            events.append(
                _event(
                    session_id=str(row["session_id"]),
                    tool="zcode",
                    event_time=event_time,
                    cwd=str(session_row["directory"] or ""),
                    model=model,
                    provider=provider,
                    input_tokens=int(row["input_tokens"] or 0),
                    cached_input_tokens=int(row["cache_read_input_tokens"] or 0),
                    cache_write_input_tokens=int(row["cache_creation_input_tokens"] or 0),
                    output_tokens=int(row["output_tokens"] or 0) + int(row["reasoning_tokens"] or 0),
                    total_tokens=int(row["computed_total_tokens"] or 0),
                    is_subagent=bool(session_row["parent_id"]),
                    tool_calls=int(row["tool_call_count"] or 0),
                    errors=int(row["status"] == "error"),
                )
            )
        for row in session_rows:
            created = _parse_time(row["time_created"])
            if not created:
                continue
            model, provider = session_models.get(str(row["id"]), ("未知模型", "ZCode"))
            sessions.append(
                _session(
                    session_id=str(row["id"]),
                    tool="zcode",
                    cwd=str(row["directory"] or ""),
                    title=str(row["title"] or "ZCode 会话"),
                    model=model,
                    provider=provider,
                    created_at=created,
                    updated_at=_parse_time(row["time_updated"]) or created,
                    is_subagent=bool(row["parent_id"]),
                    archived=bool(row["time_archived"]),
                )
            )
    return sessions, events


def _zcode_event_from_row(
    row: sqlite3.Row, session_map: dict[str, sqlite3.Row]
) -> dict[str, Any] | None:
    session_row = session_map.get(str(row["session_id"]))
    if not session_row:
        return None
    event_time = _parse_time(row["started_at"])
    if not event_time:
        return None
    model = str(row["model_id"] or "未知模型")
    return _event(
        session_id=str(row["session_id"]),
        tool="zcode",
        event_time=event_time,
        cwd=str(session_row["directory"] or ""),
        model=model,
        provider=_provider_from_model(model, "ZCode"),
        input_tokens=int(row["input_tokens"] or 0),
        cached_input_tokens=int(row["cache_read_input_tokens"] or 0),
        cache_write_input_tokens=int(row["cache_creation_input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0) + int(row["reasoning_tokens"] or 0),
        total_tokens=int(row["computed_total_tokens"] or 0),
        is_subagent=bool(session_row["parent_id"]),
        tool_calls=int(row["tool_call_count"] or 0),
        errors=int(row["status"] == "error"),
    )


def _load_zcode_cached(
    db_path: Path, cache_db: Path, now: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not db_path.is_file():
        return [], []
    source_group = str(db_path.resolve())
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as source_conn:
        source_conn.row_factory = sqlite3.Row
        usage_columns = {
            str(row["name"])
            for row in source_conn.execute("PRAGMA table_info(model_usage)").fetchall()
        }
        if "id" not in usage_columns:
            return _load_zcode(db_path)
        session_rows = source_conn.execute(
            "SELECT id, parent_id, directory, title, time_created, time_updated, time_archived FROM session"
        ).fetchall()
        session_map = {str(row["id"]): row for row in session_rows}

        cache_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(cache_db, timeout=30) as cache_conn:
            _init_external_cache(cache_conn)
            indexed = _cache_group_rows(cache_conn, "zcode", source_group)
            valid_index = {
                key: row
                for key, row in indexed.items()
                if int(row["parser_version"]) == EXTERNAL_PARSER_VERSION
            }
            if len(valid_index) != len(indexed):
                cache_conn.execute(
                    "DELETE FROM source_cache WHERE tool = 'zcode' AND source_group = ?",
                    (source_group,),
                )
                valid_index = {}
            select_fields = """
                rowid AS source_rowid, id, session_id, provider_id, model_id,
                started_at, input_tokens, output_tokens, reasoning_tokens,
                cache_creation_input_tokens, cache_read_input_tokens,
                computed_total_tokens, tool_call_count, status
            """
            terminal = "status IN ('completed', 'error', 'cancelled')"
            if valid_index:
                max_rowid = max(int(row["source_time"] or 0) for row in valid_index.values())
                hot_cutoff = int(
                    (now - timedelta(days=HOT_WINDOW_DAYS)).timestamp() * 1000
                )
                completed_clause = (
                    " OR completed_at >= ?" if "completed_at" in usage_columns else ""
                )
                params: list[int] = [hot_cutoff]
                if completed_clause:
                    params.append(hot_cutoff)
                params.append(max_rowid)
                usage_rows = source_conn.execute(
                    f"""
                    SELECT {select_fields}
                    FROM model_usage
                    WHERE {terminal}
                      AND (started_at >= ?{completed_clause} OR rowid > ?)
                    """,
                    params,
                ).fetchall()
            else:
                usage_rows = source_conn.execute(
                    f"SELECT {select_fields} FROM model_usage WHERE {terminal}"
                ).fetchall()
            for row in usage_rows:
                event = _zcode_event_from_row(row, session_map)
                if event is None:
                    continue
                _upsert_cache_item(
                    cache_conn,
                    tool="zcode",
                    source_group=source_group,
                    source_key=str(row["id"]),
                    sessions=[],
                    events=[event],
                    source_time=int(row["source_rowid"] or 0),
                )
            cache_conn.commit()
            cached_rows = _cache_group_rows(cache_conn, "zcode", source_group)

    events: list[dict[str, Any]] = []
    session_models: dict[str, tuple[str, str]] = {}
    for row in cached_rows.values():
        for event in _records_from_json(row["events_json"]):
            session_row = session_map.get(str(event.get("session_id") or ""))
            if not session_row:
                continue
            event["cwd"] = str(session_row["directory"] or "")
            event["is_subagent"] = bool(session_row["parent_id"])
            events.append(event)
            session_models[str(event["session_id"])] = (
                str(event["model"]),
                str(event["provider"]),
            )
    sessions: list[dict[str, Any]] = []
    for row in session_rows:
        created = _parse_time(row["time_created"])
        if not created:
            continue
        model, provider = session_models.get(str(row["id"]), ("未知模型", "ZCode"))
        sessions.append(
            _session(
                session_id=str(row["id"]),
                tool="zcode",
                cwd=str(row["directory"] or ""),
                title=str(row["title"] or "ZCode 会话"),
                model=model,
                provider=provider,
                created_at=created,
                updated_at=_parse_time(row["time_updated"]) or created,
                is_subagent=bool(row["parent_id"]),
                archived=bool(row["time_archived"]),
            )
        )
    return sessions, events


def _load_workbuddy(
    db_path: Path, traces_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if not db_path.is_file():
        return sessions, events
    session_map: dict[str, sqlite3.Row] = {}
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT id, cwd, title, custom_title, model, created_at, updated_at, deleted_at FROM sessions"
        ).fetchall():
            session_map[str(row["id"])] = row
    session_models: dict[str, tuple[str, str]] = {}
    if traces_root.is_dir():
        for path in traces_root.glob("*/*.json"):
            try:
                trace = (json.loads(path.read_text(encoding="utf-8", errors="replace")).get("trace") or {})
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            model_info = trace.get("modelInfo") or {}
            session_id = str(trace.get("sessionId") or "")
            session_row = session_map.get(session_id)
            event_time = _parse_time(trace.get("startedAt"))
            if not model_info or not session_row or not event_time:
                continue
            models = [str(item) for item in (model_info.get("models") or []) if item]
            model = ", ".join(models) or str(session_row["model"] or "未知模型")
            provider = _provider_from_model(models[0] if models else model, "WorkBuddy")
            session_models[session_id] = (model, provider)
            total_input = int(model_info.get("totalInputTokens") or 0)
            cached = int(model_info.get("totalCachedTokens") or 0)
            output = int(model_info.get("totalOutputTokens") or 0)
            total = int(trace.get("totalTokens") or total_input + output)
            events.append(
                _event(
                    session_id=session_id,
                    tool="workbuddy",
                    event_time=event_time,
                    cwd=str(session_row["cwd"] or ""),
                    model=model,
                    provider=provider,
                    input_tokens=total_input,
                    cached_input_tokens=cached,
                    cache_write_input_tokens=0,
                    output_tokens=output,
                    total_tokens=total,
                    is_subagent=False,
                    requests=int(model_info.get("callCount") or 0),
                    errors=int(str(trace.get("status") or "") == "error"),
                )
            )
    for session_id, row in session_map.items():
        created = _parse_time(row["created_at"])
        if not created:
            continue
        model, provider = session_models.get(
            session_id,
            (str(row["model"] or "未知模型"), _provider_from_model(str(row["model"] or ""), "WorkBuddy")),
        )
        sessions.append(
            _session(
                session_id=session_id,
                tool="workbuddy",
                cwd=str(row["cwd"] or ""),
                title=str(row["custom_title"] or row["title"] or "WorkBuddy 会话"),
                model=model,
                provider=provider,
                created_at=created,
                updated_at=_parse_time(row["updated_at"]) or created,
                is_subagent=False,
                archived=False,
            )
        )
    return sessions, events


def _load_workbuddy_trace(
    path: Path, session_map: dict[str, sqlite3.Row]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        trace = (
            json.loads(path.read_text(encoding="utf-8", errors="replace")).get("trace")
            or {}
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return [], []
    model_info = trace.get("modelInfo") or {}
    session_id = str(trace.get("sessionId") or "")
    session_row = session_map.get(session_id)
    event_time = _parse_time(trace.get("startedAt"))
    if not model_info or not session_row or not event_time:
        return [], []
    models = [str(item) for item in (model_info.get("models") or []) if item]
    model = ", ".join(models) or str(session_row["model"] or "未知模型")
    provider = _provider_from_model(models[0] if models else model, "WorkBuddy")
    total_input = int(model_info.get("totalInputTokens") or 0)
    cached = int(model_info.get("totalCachedTokens") or 0)
    output = int(model_info.get("totalOutputTokens") or 0)
    total = int(trace.get("totalTokens") or total_input + output)
    return [], [
        _event(
            session_id=session_id,
            tool="workbuddy",
            event_time=event_time,
            cwd=str(session_row["cwd"] or ""),
            model=model,
            provider=provider,
            input_tokens=total_input,
            cached_input_tokens=cached,
            cache_write_input_tokens=0,
            output_tokens=output,
            total_tokens=total,
            is_subagent=False,
            requests=int(model_info.get("callCount") or 0),
            errors=int(str(trace.get("status") or "") == "error"),
        )
    ]


def _load_workbuddy_cached(
    db_path: Path, traces_root: Path, cache_db: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not db_path.is_file():
        return [], []
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as source_conn:
        source_conn.row_factory = sqlite3.Row
        session_rows = source_conn.execute(
            "SELECT id, cwd, title, custom_title, model, created_at, updated_at, deleted_at FROM sessions"
        ).fetchall()
    session_map = {str(row["id"]): row for row in session_rows}
    source_group = f"{db_path.resolve()}::{traces_root.resolve()}"
    paths = sorted(traces_root.glob("*/*.json")) if traces_root.is_dir() else []
    events: list[dict[str, Any]] = []
    session_models: dict[str, tuple[str, str]] = {}
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_db, timeout=30) as cache_conn:
        _init_external_cache(cache_conn)
        indexed = _cache_group_rows(cache_conn, "workbuddy", source_group)
        live_keys: set[str] = set()
        for path in paths:
            source_key = str(path.resolve())
            live_keys.add(source_key)
            try:
                stat = path.stat()
            except OSError:
                continue
            prior = indexed.get(source_key)
            if (
                prior is not None
                and int(prior["size"]) == stat.st_size
                and int(prior["mtime_ns"]) == stat.st_mtime_ns
                and int(prior["parser_version"]) == EXTERNAL_PARSER_VERSION
            ):
                trace_events = _records_from_json(prior["events_json"])
            else:
                _, trace_events = _load_workbuddy_trace(path, session_map)
                _upsert_cache_item(
                    cache_conn,
                    tool="workbuddy",
                    source_group=source_group,
                    source_key=source_key,
                    sessions=[],
                    events=trace_events,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            for event in trace_events:
                session_row = session_map.get(str(event.get("session_id") or ""))
                if not session_row:
                    continue
                event["cwd"] = str(session_row["cwd"] or "")
                event["is_subagent"] = False
                events.append(event)
                session_models[str(event["session_id"])] = (
                    str(event["model"]),
                    str(event["provider"]),
                )
        stale_keys = set(indexed) - live_keys
        if stale_keys:
            cache_conn.executemany(
                "DELETE FROM source_cache WHERE tool = 'workbuddy' AND source_group = ? AND source_key = ?",
                [(source_group, source_key) for source_key in stale_keys],
            )
        cache_conn.commit()
    sessions: list[dict[str, Any]] = []
    for row in session_rows:
        created = _parse_time(row["created_at"])
        if not created:
            continue
        fallback_model = str(row["model"] or "未知模型")
        model, provider = session_models.get(
            str(row["id"]),
            (fallback_model, _provider_from_model(fallback_model, "WorkBuddy")),
        )
        sessions.append(
            _session(
                session_id=str(row["id"]),
                tool="workbuddy",
                cwd=str(row["cwd"] or ""),
                title=str(row["custom_title"] or row["title"] or "WorkBuddy 会话"),
                model=model,
                provider=provider,
                created_at=created,
                updated_at=_parse_time(row["updated_at"]) or created,
                is_subagent=False,
                archived=False,
            )
        )
    return sessions, events


def _external_payload(
    tool: str,
    sessions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    range_key: str,
    provider: str,
    now: datetime,
) -> dict[str, Any]:
    start = _range_start(range_key, now)
    start_date = start.date().isoformat() if start else None
    end_date = now.date().isoformat()
    root_sessions_all = [row for row in sessions if not row["is_subagent"]]
    providers: dict[str, dict[str, Any]] = {}
    tokens_by_provider: dict[str, int] = defaultdict(int)
    for event in events:
        if event["date"] <= end_date and (not start_date or event["date"] >= start_date):
            tokens_by_provider[event["provider"]] += event["total_tokens"]
    for row in root_sessions_all:
        if row["created_at"].date() <= now.date() and (not start or row["created_at"].date() >= start.date()):
            item = providers.setdefault(row["provider"], {"name": row["provider"], "tasks": 0, "tokens": 0})
            item["tasks"] += 1
    for name, tokens in tokens_by_provider.items():
        providers.setdefault(name, {"name": name, "tasks": 0, "tokens": 0})["tokens"] = tokens
    selected_provider = None if provider.casefold() == "all" else _provider_label(provider)
    filtered_events = [
        event for event in events
        if event["date"] <= end_date
        and (not start_date or event["date"] >= start_date)
        and (selected_provider is None or event["provider"] == selected_provider)
    ]
    filtered_sessions = [
        row for row in sessions
        if row["created_at"].date() <= now.date()
        and (not start or row["created_at"].date() >= start.date())
        and (selected_provider is None or row["provider"] == selected_provider)
    ]
    root_sessions = [row for row in filtered_sessions if not row["is_subagent"]]
    daily: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "tasks": 0, "tokens": 0, "input_tokens": 0, "cached_input_tokens": 0,
        "cache_write_input_tokens": 0, "output_tokens": 0, "root_tokens": 0,
        "subagent_tokens": 0, "requests": 0, "tool_calls": 0, "errors": 0,
        "thread_ids": set(), "root_ids": set(), "subagent_ids": set(),
    })
    for row in root_sessions:
        daily[row["created_at"].date().isoformat()]["tasks"] += 1
    projects: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for row in root_sessions:
        project = projects.setdefault(row["cwd"], {"name": _project_label(row["cwd"]), "cwd": row["cwd"], "tasks": 0, "tokens": 0})
        project["tasks"] += 1
        model = models.setdefault(row["model"], {"name": row["model"], "tasks": 0, "tokens": 0})
        model["tasks"] += 1
    project_daily: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in filtered_events:
        row = daily[event["date"]]
        for field in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "total_tokens"):
            target = "tokens" if field == "total_tokens" else field
            row[target] += event[field]
        row["requests"] += event["requests"]
        row["tool_calls"] += event["tool_calls"]
        row["errors"] += event["errors"]
        row["thread_ids"].add(event["session_id"])
        agent_key = "subagent_tokens" if event["is_subagent"] else "root_tokens"
        id_key = "subagent_ids" if event["is_subagent"] else "root_ids"
        row[agent_key] += event["total_tokens"]
        row[id_key].add(event["session_id"])
        project = projects.setdefault(event["cwd"], {"name": _project_label(event["cwd"]), "cwd": event["cwd"], "tasks": 0, "tokens": 0})
        project["tokens"] += event["total_tokens"]
        model = models.setdefault(event["model"], {"name": event["model"], "tasks": 0, "tokens": 0})
        model["tokens"] += event["total_tokens"]
        day_project = project_daily[event["date"]].setdefault(event["cwd"], {"name": _project_label(event["cwd"]), "cwd": event["cwd"], "tasks": set(), "tokens": 0})
        day_project["tasks"].add(event["session_id"])
        day_project["tokens"] += event["total_tokens"]
    dates = set(daily)
    if range_key == "all":
        first_day = min((datetime.strptime(value, "%Y-%m-%d").date() for value in dates), default=None)
    else:
        first_day = start.date() if start else None
    daily_rows: list[dict[str, Any]] = []
    if first_day:
        cursor = first_day
        while cursor <= now.date():
            date_key = cursor.isoformat()
            row = daily[date_key]
            input_tokens = int(row["input_tokens"])
            total_tokens = int(row["tokens"])
            involved = len(row["thread_ids"])
            daily_rows.append({
                "date": date_key, "tasks": int(row["tasks"]), "tokens": total_tokens,
                "input_tokens": input_tokens, "cached_input_tokens": int(row["cached_input_tokens"]),
                "cache_write_input_tokens": int(row["cache_write_input_tokens"]),
                "non_cached_input_tokens": max(0, input_tokens - int(row["cached_input_tokens"])),
                "output_tokens": int(row["output_tokens"]),
                "unclassified_tokens": max(0, total_tokens - input_tokens - int(row["output_tokens"])),
                "involved_threads": involved, "root_threads": len(row["root_ids"]),
                "subagent_threads": len(row["subagent_ids"]), "root_tokens": int(row["root_tokens"]),
                "subagent_tokens": int(row["subagent_tokens"]),
                "tokens_per_thread": round(total_tokens / involved) if involved else 0,
                "cache_ratio": round(int(row["cached_input_tokens"]) / input_tokens, 4) if input_tokens else 0,
                "requests": int(row["requests"]), "tool_calls": int(row["tool_calls"]), "errors": int(row["errors"]),
            })
            cursor += timedelta(days=1)
    totals = {field: sum(int(row[field]) for row in daily_rows) for field in ("tokens", "input_tokens", "cached_input_tokens", "output_tokens", "root_tokens", "subagent_tokens", "requests", "tool_calls", "errors")}
    involved_ids = {event["session_id"] for event in filtered_events if event["total_tokens"] > 0}
    project_rows = sorted(projects.values(), key=lambda item: (-item["tokens"], -item["tasks"]))
    return {
        "range": range_key, "provider": selected_provider or "all", "provider_label": selected_provider or "全部 Provider",
        "summary": {"tasks": len(root_sessions), "subagent_tasks": len(filtered_sessions) - len(root_sessions), "tokens": totals["tokens"],
            "input_tokens": totals["input_tokens"], "cached_input_tokens": totals["cached_input_tokens"],
            "non_cached_input_tokens": max(0, totals["input_tokens"] - totals["cached_input_tokens"]),
            "output_tokens": totals["output_tokens"], "cache_ratio": round(totals["cached_input_tokens"] / totals["input_tokens"], 4) if totals["input_tokens"] else 0,
            "active_days": sum(1 for row in daily_rows if row["tokens"] > 0), "projects": len(project_rows),
            "archived_tasks": sum(int(row["archived"]) for row in root_sessions), "involved_threads": len(involved_ids),
            "root_tokens": totals["root_tokens"], "subagent_tokens": totals["subagent_tokens"], "requests": totals["requests"],
            "tool_calls": totals["tool_calls"], "errors": totals["errors"], "first_date": min(dates) if dates else None, "last_date": max(dates) if dates else None},
        "daily": daily_rows, "projects": project_rows[:12], "models": sorted(models.values(), key=lambda item: (-item["tokens"], -item["tasks"])),
        "providers": sorted(providers.values(), key=lambda item: (-item["tokens"], -item["tasks"])),
        "project_daily": {date: sorted(({**item, "tasks": len(item["tasks"])} for item in values.values()), key=lambda item: (-item["tokens"], -item["tasks"])) for date, values in project_daily.items()},
        "recent_tasks": [{"title": row["title"], "project": _project_label(row["cwd"]), "cwd": row["cwd"], "model": row["model"], "provider": row["provider"], "tokens": sum(event["total_tokens"] for event in filtered_events if event["session_id"] == row["id"]), "created_at": row["created_at"].isoformat(timespec="seconds"), "updated_at": row["updated_at"].isoformat(timespec="seconds"), "archived": row["archived"], "tool": TOOL_LABELS[tool]} for row in sorted(root_sessions, key=lambda item: item["created_at"], reverse=True)[:20]],
    }


def _codex_payload(
    range_key: str,
    provider: str,
    now: datetime,
    *,
    state_db: str | Path | None = None,
    cache_db: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_codex_usage_stats(
        range_key=range_key,
        provider=provider,
        now=now,
        **({"state_db": state_db} if state_db is not None else {}),
        **({"cache_db": cache_db} if cache_db is not None else {}),
    )
    for row in payload["daily"]:
        row.setdefault("requests", 0); row.setdefault("tool_calls", 0); row.setdefault("errors", 0)
    for row in payload["recent_tasks"]:
        row["tool"] = "Codex"
    payload["summary"].setdefault("requests", 0); payload["summary"].setdefault("tool_calls", 0); payload["summary"].setdefault("errors", 0)
    return payload


def _merge_payloads(payloads: list[tuple[str, dict[str, Any]]], range_key: str, provider: str, tool: str, now: datetime) -> dict[str, Any]:
    daily_map: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    projects: dict[str, dict[str, Any]] = {}
    project_daily: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    models: dict[str, dict[str, Any]] = {}
    providers: dict[str, dict[str, Any]] = {}
    recent: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    summary_fields = ("tasks", "subagent_tasks", "tokens", "input_tokens", "cached_input_tokens", "non_cached_input_tokens", "output_tokens", "projects", "archived_tasks", "involved_threads", "root_tokens", "subagent_tokens", "requests", "tool_calls", "errors")
    summary = {field: 0 for field in summary_fields}
    for tool_key, payload in payloads:
        current = payload["summary"]
        for field in summary_fields:
            if field != "projects": summary[field] += int(current.get(field) or 0)
        tool_rows.append({"key": tool_key, "name": TOOL_LABELS[tool_key], "tasks": int(current.get("tasks") or 0), "subagent_tasks": int(current.get("subagent_tasks") or 0), "tokens": int(current.get("tokens") or 0), "active_days": int(current.get("active_days") or 0), "cache_ratio": float(current.get("cache_ratio") or 0), "data_quality": "完整" if tool_key != "workbuddy" else "Token 完整，子 Agent 拆分不可用"})
        for row in payload["daily"]:
            target = daily_map[row["date"]]
            target["date"] = row["date"]
            for field in ("tasks", "tokens", "input_tokens", "cached_input_tokens", "cache_write_input_tokens", "non_cached_input_tokens", "output_tokens", "unclassified_tokens", "involved_threads", "root_threads", "subagent_threads", "root_tokens", "subagent_tokens", "requests", "tool_calls", "errors"):
                target[field] += int(row.get(field) or 0)
            target.setdefault("tool_tokens", {})[tool_key] = int(row.get("tokens") or 0)
        for row in payload.get("projects", []):
            target = projects.setdefault(row.get("cwd") or row["name"], {"name": row["name"], "cwd": row.get("cwd", ""), "tasks": 0, "tokens": 0})
            target["tasks"] += int(row.get("tasks") or 0); target["tokens"] += int(row.get("tokens") or 0)
        for date, rows in payload.get("project_daily", {}).items():
            for row in rows:
                key = row.get("cwd") or row["name"]
                target = project_daily[date].setdefault(key, {"name": row["name"], "cwd": row.get("cwd", ""), "tasks": 0, "tokens": 0})
                target["tasks"] += int(row.get("tasks") or 0); target["tokens"] += int(row.get("tokens") or 0)
        for row in payload.get("models", []):
            target = models.setdefault(row["name"], {"name": row["name"], "tasks": 0, "tokens": 0})
            target["tasks"] += int(row.get("tasks") or 0); target["tokens"] += int(row.get("tokens") or 0)
        for row in payload.get("providers", []):
            target = providers.setdefault(row["name"], {"name": row["name"], "tasks": 0, "tokens": 0})
            target["tasks"] += int(row.get("tasks") or 0); target["tokens"] += int(row.get("tokens") or 0)
        recent.extend(payload.get("recent_tasks", []))
    daily_rows = []
    for date in sorted(daily_map):
        row = dict(daily_map[date]); involved = int(row.get("involved_threads") or 0); input_tokens = int(row.get("input_tokens") or 0)
        row["tokens_per_thread"] = round(int(row.get("tokens") or 0) / involved) if involved else 0
        row["cache_ratio"] = round(int(row.get("cached_input_tokens") or 0) / input_tokens, 4) if input_tokens else 0
        daily_rows.append(row)
    summary["active_days"] = sum(1 for row in daily_rows if row.get("tokens", 0) > 0)
    summary["projects"] = len(projects)
    summary["cache_ratio"] = round(summary["cached_input_tokens"] / summary["input_tokens"], 4) if summary["input_tokens"] else 0
    active_dates = [row["date"] for row in daily_rows if row.get("tokens", 0) > 0 or row.get("tasks", 0) > 0]
    summary["first_date"] = min(active_dates) if active_dates else None; summary["last_date"] = max(active_dates) if active_dates else None
    tool_days = [sum(1 for value in row.get("tool_tokens", {}).values() if value > 0) for row in daily_rows if row.get("tokens", 0) > 0]
    multi_tool_days = sum(1 for count in tool_days if count > 1)
    dominant = max(tool_rows, key=lambda item: item["tokens"], default=None)
    total_tokens = int(summary["tokens"])
    insights = [
        {"label": "多工具切换", "value": f"{multi_tool_days}/{len(tool_days)} 个活跃日", "level": "warn" if tool_days and multi_tool_days / len(tool_days) > 0.5 else "ok", "detail": "同一天使用两种以上 AI 工具；比例过高通常意味着上下文在工具间重复搬运。"},
        {"label": "工具集中度", "value": f"{dominant['name']} {dominant['tokens'] / total_tokens:.0%}" if dominant and total_tokens else "暂无", "level": "info", "detail": "处理量最高工具占全部 Token 的比例，用于判断主力工具是否稳定。"},
        {"label": "子 Agent 占比", "value": f"{summary['subagent_tokens'] / total_tokens:.1%}" if total_tokens else "0%", "level": "warn" if total_tokens and summary["subagent_tokens"] / total_tokens > 0.4 else "ok", "detail": "仅统计能够可靠识别子 Agent Token 的工具；WorkBuddy 暂不参与该拆分。"},
        {"label": "平均每活跃会话", "value": f"{round(total_tokens / summary['involved_threads']):,} Token" if summary["involved_threads"] else "0", "level": "info", "detail": "数值持续升高通常表示会话过长、上下文膨胀或任务边界不清。"},
    ]
    return {
        "range": range_key, "range_label": "全部" if range_key == "all" else f"近 {range_key} 日", "provider": provider, "provider_label": "全部 Provider" if provider.casefold() == "all" else _provider_label(provider),
        "tool": tool, "tool_label": "全部 AI 工具" if tool == "all" else TOOL_LABELS[tool], "generated_at": now.isoformat(timespec="seconds"),
        "methodology": "会话按创建日；Token 按各工具本地 usage 事件归入发生日（北京时间）。跨工具 Token 只代表处理工作量，不等同于费用。",
        "summary": summary, "daily": daily_rows, "projects": sorted(projects.values(), key=lambda item: (-item["tokens"], -item["tasks"]))[:12],
        "project_daily": {date: sorted(rows.values(), key=lambda item: (-item["tokens"], -item["tasks"])) for date, rows in project_daily.items()},
        "models": sorted(models.values(), key=lambda item: (-item["tokens"], -item["tasks"])), "providers": sorted(providers.values(), key=lambda item: (-item["tokens"], -item["tasks"])),
        "tools": sorted(tool_rows, key=lambda item: -item["tokens"]), "insights": insights,
        "recent_tasks": sorted(recent, key=lambda item: item.get("created_at", ""), reverse=True)[:20],
    }


def build_ai_usage_stats(
    *,
    range_key: str = "30",
    provider: str = "all",
    tool: str = "all",
    now: datetime | None = None,
    claude_root: str | Path = DEFAULT_CLAUDE_ROOT,
    zcode_db: str | Path = DEFAULT_ZCODE_DB,
    workbuddy_db: str | Path = DEFAULT_WORKBUDDY_DB,
    workbuddy_traces: str | Path = DEFAULT_WORKBUDDY_TRACES,
    usage_cache_db: str | Path = DEFAULT_AI_USAGE_CACHE_DB,
    codex_state_db: str | Path | None = None,
    gateway_db: str | Path | None = None,
    cache_ttl_seconds: float = 15,
) -> dict[str, Any]:
    if range_key not in ALLOWED_RANGES:
        raise ValueError(f"unsupported range: {range_key}")
    tool_key = str(tool or "all").strip().casefold()
    if tool_key not in ALLOWED_TOOLS:
        raise ValueError(f"unsupported tool: {tool}")
    now = now or datetime.now(LOCAL_TZ)
    cache_db = Path(usage_cache_db).expanduser()
    selected = list(TOOL_LABELS) if tool_key == "all" else [tool_key]
    # Codex 库不存在时跳过：其它三个数据源在缺文件时都是返回空，这里保持一致，
    # 否则一台没装 Codex 的机器会整个页面报错。
    def _codex_available() -> bool:
        if codex_state_db is None:
            return default_codex_state_db().is_file()
        return Path(codex_state_db).expanduser().is_file()

    payloads: list[tuple[str, dict[str, Any]]] = []
    if "codex" in selected and _codex_available():
        payloads.append(
            (
                "codex",
                _codex_payload(
                    range_key, provider, now, state_db=codex_state_db, cache_db=usage_cache_db
                ),
            )
        )
    if "claude" in selected:
        root = Path(claude_root).expanduser()
        sessions, events = _cached_source(
            f"claude:{root}:{cache_db}",
            lambda: _load_claude_cached(root, cache_db),
            cache_ttl_seconds,
        )
        payloads.append(("claude", _external_payload("claude", sessions, events, range_key=range_key, provider=provider, now=now)))
    if "zcode" in selected:
        path = Path(zcode_db).expanduser()
        sessions, events = _cached_source(
            f"zcode:{path}:{cache_db}",
            lambda: _load_zcode_cached(path, cache_db, now),
            cache_ttl_seconds,
        )
        payloads.append(("zcode", _external_payload("zcode", sessions, events, range_key=range_key, provider=provider, now=now)))
    if "workbuddy" in selected:
        db_path, traces_path = Path(workbuddy_db).expanduser(), Path(workbuddy_traces).expanduser()
        sessions, events = _cached_source(
            f"workbuddy:{db_path}:{traces_path}:{cache_db}",
            lambda: _load_workbuddy_cached(db_path, traces_path, cache_db),
            cache_ttl_seconds,
        )
        payloads.append(("workbuddy", _external_payload("workbuddy", sessions, events, range_key=range_key, provider=provider, now=now)))
    resolved_gateway_db = gateway_db if gateway_db is not None else default_gateway_db()
    if "gateway" in selected and resolved_gateway_db is not None:
        from buddy2api.ai_usage.gateway_source import load_gateway

        path = Path(resolved_gateway_db).expanduser()
        # 只读 range 起点之后的日志：logs 表会无限增长，全表扫会越来越慢
        start = _range_start(range_key, now)
        since = (start or now - timedelta(days=365 * 5)).astimezone(LOCAL_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sessions, events = _cached_source(
            f"gateway:{path}:{since.isoformat()}",
            lambda: load_gateway(path, since=since),
            cache_ttl_seconds,
        )
        payloads.append(("gateway", _external_payload("gateway", sessions, events, range_key=range_key, provider=provider, now=now)))
    return _merge_payloads(payloads, range_key, provider, tool_key, now)
