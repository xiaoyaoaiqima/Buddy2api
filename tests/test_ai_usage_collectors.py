import json
import sqlite3
from datetime import datetime

import buddy2api.ai_usage.collectors as ai_usage_stats
from buddy2api.ai_usage.collectors import (
    _load_claude_cached,
    _load_workbuddy,
    _load_workbuddy_cached,
    _load_zcode,
    _load_zcode_cached,
    build_ai_usage_stats,
)


def test_build_ai_usage_stats_reads_claude_root_and_subagent_usage(tmp_path):
    project = tmp_path / "claude" / "-Users-test-project"
    project.mkdir(parents=True)
    root_event = {
        "timestamp": "2026-09-02T02:00:00Z",
        "cwd": "/Users/test/project",
        "slug": "root-session",
        "message": {
            "id": "message-root",
            "model": "claude-test",
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 20,
                "output_tokens": 10,
            },
            "content": [{"type": "tool_use"}],
        },
    }
    child_event = {
        "timestamp": "2026-09-02T03:00:00Z",
        "cwd": "/Users/test/project",
        "message": {
            "id": "message-child",
            "model": "claude-test",
            "usage": {
                "input_tokens": 5,
                "cache_read_input_tokens": 25,
                "output_tokens": 5,
            },
        },
    }
    (project / "root.jsonl").write_text(json.dumps(root_event) + "\n")
    (project / "agent-child.jsonl").write_text(json.dumps(child_event) + "\n")

    result = build_ai_usage_stats(
        range_key="7",
        tool="claude",
        now=datetime(2026, 9, 3, 9),
        claude_root=tmp_path / "claude",
        usage_cache_db=tmp_path / "usage-cache.sqlite",
        cache_ttl_seconds=0,
    )

    assert result["summary"]["tasks"] == 1
    assert result["summary"]["subagent_tasks"] == 1
    assert result["summary"]["tokens"] == 135
    assert result["summary"]["cached_input_tokens"] == 85
    assert result["summary"]["root_tokens"] == 100
    assert result["summary"]["subagent_tokens"] == 35
    assert result["summary"]["tool_calls"] == 1
    assert result["tools"][0]["name"] == "Claude Code"


def test_load_zcode_uses_model_usage_and_parent_session(tmp_path):
    db = tmp_path / "zcode.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE session (
                id TEXT, parent_id TEXT, directory TEXT, title TEXT,
                time_created INTEGER, time_updated INTEGER, time_archived INTEGER
            );
            CREATE TABLE model_usage (
                session_id TEXT, provider_id TEXT, model_id TEXT, started_at INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
                cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER,
                computed_total_tokens INTEGER, tool_call_count INTEGER, status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO session VALUES (?,?,?,?,?,?,?)",
            ("child", "root", "/Users/test/zcode", "child", 1788314400000, 1788314400000, None),
        )
        conn.execute(
            "INSERT INTO model_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("child", "builtin", "glm-test", 1788314400000, 100, 10, 2, 0, 80, 112, 3, "completed"),
        )

    sessions, events = _load_zcode(db)

    assert sessions[0]["is_subagent"] is True
    assert events[0]["cached_input_tokens"] == 80
    assert events[0]["output_tokens"] == 12
    assert events[0]["total_tokens"] == 112
    assert events[0]["tool_calls"] == 3


def test_load_workbuddy_reads_trace_usage_without_fake_subagent_split(tmp_path):
    db = tmp_path / "workbuddy.sqlite"
    traces = tmp_path / "traces" / "123"
    traces.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT, cwd TEXT, title TEXT, custom_title TEXT, model TEXT,
                created_at INTEGER, updated_at INTEGER, deleted_at INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
            ("session-1", "/Users/test/work", "Work", None, "deepseek-test", 1788314400000, 1788314400000, None),
        )
    (traces / "trace.json").write_text(
        json.dumps(
            {
                "trace": {
                    "sessionId": "session-1",
                    "startedAt": 1788314400000,
                    "status": "ok",
                    "totalTokens": 120,
                    "modelInfo": {
                        "models": ["deepseek-test"],
                        "totalInputTokens": 100,
                        "totalCachedTokens": 70,
                        "totalOutputTokens": 20,
                        "callCount": 2,
                    },
                }
            }
        )
    )

    sessions, events = _load_workbuddy(db, tmp_path / "traces")

    assert sessions[0]["is_subagent"] is False
    assert events[0]["total_tokens"] == 120
    assert events[0]["cached_input_tokens"] == 70
    assert events[0]["requests"] == 2
    assert events[0]["is_subagent"] is False


def test_claude_persistent_cache_reuses_unchanged_jsonl(tmp_path, monkeypatch):
    root = tmp_path / "claude"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-02T02:00:00Z",
                "cwd": "/Users/test/project",
                "message": {
                    "id": "message-1",
                    "model": "claude-test",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            }
        )
        + "\n"
    )
    cache_db = tmp_path / "usage-cache.sqlite"

    first = _load_claude_cached(root, cache_db)
    monkeypatch.setattr(
        ai_usage_stats,
        "_load_claude_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("unchanged file reparsed")),
    )
    second = _load_claude_cached(root, cache_db)

    assert second == first


def test_zcode_cache_only_reloads_hot_or_new_usage_rows(tmp_path, monkeypatch):
    db = tmp_path / "zcode.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE session (
                id TEXT, parent_id TEXT, directory TEXT, title TEXT,
                time_created INTEGER, time_updated INTEGER, time_archived INTEGER
            );
            CREATE TABLE model_usage (
                id TEXT PRIMARY KEY, session_id TEXT, provider_id TEXT, model_id TEXT,
                started_at INTEGER, completed_at INTEGER, input_tokens INTEGER,
                output_tokens INTEGER, reasoning_tokens INTEGER,
                cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER,
                computed_total_tokens INTEGER, tool_call_count INTEGER, status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO session VALUES (?,?,?,?,?,?,?)",
            ("root", None, "/Users/test/zcode", "root", 1785636000000, 1785636000000, None),
        )
        conn.execute(
            "INSERT INTO model_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("old", "root", "builtin", "glm-test", 1785636000000, 1785636060000, 100, 10, 0, 0, 80, 110, 1, "completed"),
        )
    cache_db = tmp_path / "usage-cache.sqlite"
    now = datetime(2026, 9, 3, 9)
    _, first_events = _load_zcode_cached(db, cache_db, now)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO model_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("new", "root", "builtin", "glm-test", 1788397200000, 1788397260000, 20, 2, 0, 0, 10, 22, 1, "completed"),
        )
    parsed_ids: list[str] = []
    original = ai_usage_stats._zcode_event_from_row

    def track_row(row, session_map):
        parsed_ids.append(str(row["id"]))
        return original(row, session_map)

    monkeypatch.setattr(ai_usage_stats, "_zcode_event_from_row", track_row)
    _, second_events = _load_zcode_cached(db, cache_db, now)

    assert len(first_events) == 1
    assert {event["total_tokens"] for event in second_events} == {110, 22}
    assert parsed_ids == ["new"]


def test_workbuddy_persistent_cache_reuses_unchanged_trace(tmp_path, monkeypatch):
    db = tmp_path / "workbuddy.sqlite"
    traces = tmp_path / "traces" / "123"
    traces.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT, cwd TEXT, title TEXT, custom_title TEXT, model TEXT,
                created_at INTEGER, updated_at INTEGER, deleted_at INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
            ("session-1", "/Users/test/work", "Work", None, "deepseek-test", 1788314400000, 1788314400000, None),
        )
    (traces / "trace.json").write_text(
        json.dumps(
            {
                "trace": {
                    "sessionId": "session-1",
                    "startedAt": 1788314400000,
                    "totalTokens": 120,
                    "modelInfo": {"models": ["deepseek-test"], "callCount": 1},
                }
            }
        )
    )
    cache_db = tmp_path / "usage-cache.sqlite"

    first = _load_workbuddy_cached(db, tmp_path / "traces", cache_db)
    monkeypatch.setattr(
        ai_usage_stats,
        "_load_workbuddy_trace",
        lambda _path, _sessions: (_ for _ in ()).throw(
            AssertionError("unchanged trace reparsed")
        ),
    )
    second = _load_workbuddy_cached(db, tmp_path / "traces", cache_db)

    assert second == first
