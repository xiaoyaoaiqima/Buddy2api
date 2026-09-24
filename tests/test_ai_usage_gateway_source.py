"""网关日志数据源（`gateway_source`）的测试。

这个数据源把网关自己的 `logs` 表映射成与其它工具同构的 (sessions, events)，
所以最容易错的就是**口径**：Token 字段的含义必须和 collectors 的约定一致，
否则数字看起来"有值"但语义是错的。

已经踩过一次：collectors 用 `non_cached = input - cached` 自行相减，所以
input 必须是**含缓存的完整 prompt**。第一版在这里又减了一次 cached，
结果非缓存输入恒为 0，而总数、缓存数都正常——这种错不写测试很难发现。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from buddy2api.ai_usage import collectors
from buddy2api.ai_usage.gateway_source import GATEWAY_CWD, load_gateway


def _make_db(tmp_path, rows):
    path = tmp_path / "gw.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id INTEGER, api_key_name TEXT,
            account_id INTEGER, account_name TEXT,
            model TEXT, stream INTEGER,
            prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0, credit REAL DEFAULT 0,
            cached_tokens INTEGER DEFAULT 0, finish_reason TEXT,
            duration_ms INTEGER, status_code INTEGER, error_msg TEXT,
            created_at INTEGER
        )
        """
    )
    for r in rows:
        conn.execute(
            """INSERT INTO logs (api_key_id, api_key_name, model, prompt_tokens,
               completion_tokens, total_tokens, cached_tokens, status_code, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                r.get("key_id"), r.get("key_name"), r.get("model", "m1"),
                r["prompt"], r.get("completion", 0), r.get("total"), r.get("cached", 0),
                r.get("status", 200), int(r["at"].timestamp()),
            ),
        )
    conn.commit()
    conn.close()
    return path


def _now():
    return datetime.now(timezone.utc)


def test_missing_db_is_not_an_error(tmp_path):
    """网关库不存在时返回空：AI 用量页不该因为缺一个数据源就整个报错。"""
    sessions, events = load_gateway(tmp_path / "nope.db")
    assert sessions == [] and events == []


def test_one_session_per_api_key(tmp_path):
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": 1, "key_name": "alpha", "at": now, "prompt": 100, "completion": 10, "total": 110},
        {"key_id": 1, "key_name": "alpha", "at": now, "prompt": 200, "completion": 20, "total": 220},
        {"key_id": 2, "key_name": "beta", "at": now, "prompt": 50, "completion": 5, "total": 55},
    ])
    sessions, events = load_gateway(db)
    assert {s["id"] for s in sessions} == {"gateway:1", "gateway:2"}
    assert len(events) == 3
    assert all(s["tool"] == "gateway" for s in sessions)
    assert all(s["cwd"] == GATEWAY_CWD for s in sessions)
    titles = {s["id"]: s["title"] for s in sessions}
    assert titles["gateway:1"] == "alpha" and titles["gateway:2"] == "beta"


def test_input_tokens_include_cached(tmp_path):
    """口径测试：input 必须是含缓存的完整 prompt，由 collectors 去减 cached。

    如果这里提前减掉，collectors 再减一次，非缓存输入就会恒为 0。
    """
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": 1, "key_name": "k", "at": now, "prompt": 1000, "cached": 900,
         "completion": 50, "total": 1050},
    ])
    _, events = load_gateway(db)
    ev = events[0]
    assert ev["input_tokens"] == 1000, "input 被提前减掉了 cached"
    assert ev["cached_input_tokens"] == 900
    assert ev["output_tokens"] == 50
    assert ev["total_tokens"] == 1050


def test_non_cached_is_positive_after_aggregation(tmp_path):
    """端到端把这次踩过的坑钉死：非缓存输入不能被算成 0。"""
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": 1, "key_name": "k", "at": now, "prompt": 1000, "cached": 900,
         "completion": 50, "total": 1050},
    ])
    sessions, events = load_gateway(db)
    payload = collectors._external_payload(
        "gateway", sessions, events, range_key="30", provider="all", now=now.astimezone(collectors.LOCAL_TZ)
    )
    s = payload["summary"]
    assert s["non_cached_input_tokens"] == 100
    assert s["cached_input_tokens"] == 900
    assert s["tokens"] == 1050


def test_total_falls_back_to_prompt_plus_completion(tmp_path):
    """老日志可能没写 total_tokens，用 prompt+completion 兜底，别让用量凭空少一块。"""
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": 1, "key_name": "k", "at": now, "prompt": 30, "completion": 7, "total": 0},
    ])
    _, events = load_gateway(db)
    assert events[0]["total_tokens"] == 37


def test_errors_counted_from_status_code(tmp_path):
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": 1, "key_name": "k", "at": now, "prompt": 10, "completion": 1, "total": 11, "status": 200},
        {"key_id": 1, "key_name": "k", "at": now, "prompt": 10, "completion": 0, "total": 10, "status": 500},
    ])
    sessions, events = load_gateway(db)
    assert sum(e["errors"] for e in events) == 1
    assert sum(e["requests"] for e in events) == 2


def test_since_filters_old_rows(tmp_path):
    """since 过滤必须生效：logs 会无限增长，全表扫会越来越慢。"""
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": 1, "key_name": "k", "at": now - timedelta(days=40), "prompt": 10, "total": 10},
        {"key_id": 1, "key_name": "k", "at": now - timedelta(days=1), "prompt": 20, "total": 20},
    ])
    _, events = load_gateway(db, since=now - timedelta(days=7))
    assert len(events) == 1
    assert events[0]["input_tokens"] == 20


def test_anonymous_requests_get_a_bucket(tmp_path):
    """没鉴权的请求也要有归属，否则用量凭空消失。"""
    now = _now()
    db = _make_db(tmp_path, [
        {"key_id": None, "key_name": None, "at": now, "prompt": 10, "total": 10},
    ])
    sessions, events = load_gateway(db)
    assert len(sessions) == 1 and len(events) == 1
    assert "anon" in sessions[0]["id"]


def test_gateway_registered_as_a_tool():
    assert "gateway" in collectors.TOOL_LABELS
    assert "gateway" in collectors.ALLOWED_TOOLS


# ── 数据源缺失时的降级行为 ──
# 这个模块会被别人在自己的机器上跑，他们未必装了全部 5 个工具。
# 缺数据源必须是"少一块数据"，而不是整页报错。

def test_missing_sources_degrade_gracefully(tmp_path):
    """所有数据源都不存在时不抛异常，返回空统计。"""
    from buddy2api.ai_usage.collectors import build_ai_usage_stats

    data = build_ai_usage_stats(
        range_key="30",
        claude_root=tmp_path / "no-claude",
        zcode_db=tmp_path / "no-zcode.sqlite",
        workbuddy_db=tmp_path / "no-wb.sqlite",
        workbuddy_traces=tmp_path / "no-traces",
        gateway_db=tmp_path / "no-gw.sqlite",
        codex_state_db=tmp_path / "no-codex.sqlite",
        usage_cache_db=tmp_path / "cache.sqlite",
    )
    assert data["summary"]["tasks"] == 0
    assert data["summary"]["tokens"] == 0


def test_codex_path_is_overridable(tmp_path):
    """Codex 采集器必须接受路径覆盖。

    它原先在 _codex_payload 里硬编码读真实 ~/.codex，导致任何"换个目录跑"的
    测试或部署都会悄悄读到宿主机数据——数字看着正常，其实来源是错的。
    """
    import inspect

    from buddy2api.ai_usage import collectors

    sig = inspect.signature(collectors.build_ai_usage_stats)
    assert "codex_state_db" in sig.parameters, "build_ai_usage_stats 没有 codex_state_db 参数"

    helper = inspect.getsource(collectors._codex_payload)
    assert "state_db" in helper, "_codex_payload 没有把 state_db 透传下去"
