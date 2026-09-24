"""把网关自己的请求日志（`logs` 表）接成 AI 用量统计的第 5 个数据源。

其它四个数据源（Codex / Claude Code / ZCode / WorkBuddy）读的是**客户端**在本机留下的
轨迹；这个读的是**经过本网关**的请求。两者互补：

- 客户端轨迹能看到"我在哪个项目、哪个会话里用了多少"，但看不到直接打 API 的调用；
- 网关日志能看到全部转发流量（含按 key 的归属、缓存命中、耗时、报错），
  但看不到客户端侧的会话结构。

所以这里把网关日志映射成与其它数据源同构的 `(sessions, events)`：
**一个 api_key 一条"会话"**。理由是这样能落进现有聚合逻辑而不用改它，同时在
项目/模型分布里表现为"哪个 key 在用、用的什么模型"——这正是网关视角下最自然的切分。
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buddy2api.ai_usage.collectors import _event, _session

# 网关自己没有"项目"概念，用这个占位名归入项目分布，避免和其它工具的真实目录混在一起
GATEWAY_CWD = "（网关直连）"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def load_gateway(
    db_path: str | Path,
    *,
    since: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读 `logs` 表，产出 (sessions, events)。

    `since` 为今天零点的本地时间；只读该时间之后的日志，避免为了一个 30 天视图
    去扫全表（logs 会随时间无限增长）。
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        return [], []
    conn = _connect(path)
    try:
        params: list[Any] = []
        where = ""
        if since is not None:
            where = " WHERE created_at >= ?"
            params.append(int(since.timestamp()))
        rows = conn.execute(
            f"""
            SELECT api_key_id, api_key_name, model, prompt_tokens, completion_tokens,
                   total_tokens, cached_tokens, duration_ms, status_code, created_at
            FROM logs{where}
            ORDER BY created_at ASC
            """,
            params,
        ).fetchall()
    except sqlite3.Error:
        return [], []
    finally:
        conn.close()

    sessions: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    # 每个 key 一条会话；同时累计它的活跃区间，用于 updated_at
    bounds: dict[str, list[datetime]] = defaultdict(list)

    for row in rows:
        created = datetime.fromtimestamp(int(row["created_at"] or 0), tz=timezone.utc)
        key = f"gateway:{row['api_key_id'] if row['api_key_id'] is not None else 'anon'}"
        label = str(row["api_key_name"] or "").strip() or (
            f"API Key #{row['api_key_id']}" if row["api_key_id"] is not None else "未鉴权请求"
        )
        model = str(row["model"] or "").strip() or "未知模型"

        if key not in sessions:
            sessions[key] = _session(
                session_id=key,
                tool="gateway",
                cwd=GATEWAY_CWD,
                title=label,
                model=model,
                provider="gateway",
                created_at=created,
                updated_at=created,
                is_subagent=False,
            )
        bounds[key].append(created)

        prompt = int(row["prompt_tokens"] or 0)
        cached = int(row["cached_tokens"] or 0)
        completion = int(row["completion_tokens"] or 0)
        total = int(row["total_tokens"] or 0) or (prompt + completion)
        status = int(row["status_code"] or 0)
        events.append(
            _event(
                session_id=key,
                tool="gateway",
                event_time=created,
                cwd=GATEWAY_CWD,
                model=model,
                provider="gateway",
                # 口径对齐：collectors 用 `non_cached = input - cached` 自行相减，
                # 所以 input 必须是**含缓存的完整 prompt**。网关日志里
                # total = prompt + completion 且 prompt 已含 cached，直接透传即可。
                # （曾经在这里又减了一次 cached，结果非缓存输入恒为 0。）
                input_tokens=prompt,
                cached_input_tokens=cached,
                cache_write_input_tokens=0,
                output_tokens=completion,
                total_tokens=total,
                is_subagent=False,
                requests=1,
                errors=1 if status >= 400 else 0,
            )
        )

    for key, times in bounds.items():
        if key in sessions:
            sessions[key]["updated_at"] = max(times)
    return list(sessions.values()), events
