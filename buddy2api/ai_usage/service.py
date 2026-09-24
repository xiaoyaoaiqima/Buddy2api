"""AI 用量统计的接口层。

数据来自 `buddy2api.ai_usage.collectors`，它要扫描本机若干 GB 的客户端轨迹
（实测冷启动 ~25s，命中它自己的 SQLite 缓存后 ~0.03s）。这个数量级决定了三件事：

1. **必须在线程池里跑**：`build_ai_usage_stats` 是同步阻塞的，直接放进 async 路由
   会把整个网关的事件循环卡住 25s——那时所有转发请求都会停摆。
2. **同一时刻只能有一次扫描**：用户连点几下刷新不该触发 N 次全盘扫描。
3. **要让前端知道"这次是冷的"**：冷启动时如实返回 `warming=true`，前端才能显示
   "正在扫描本机轨迹"，而不是让人对着转圈猜是不是坏了。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from buddy2api.ai_usage import collectors

# 冷扫描在后台预热，避免第一个打开页面的人等 25s
_prewarm_task: asyncio.Task | None = None
# 扫描是同步重活：用一把锁保证同一时刻只有一次全盘扫描在跑
_scan_lock = asyncio.Lock()
_last_result: dict[str, Any] | None = None
_last_scan_at: float = 0.0
_scans = 0


def _scan(range_key: str, provider: str, tool: str) -> dict[str, Any]:
    """同步执行（在线程池里被调用）。"""
    return collectors.build_ai_usage_stats(range_key=range_key, provider=provider, tool=tool)


async def overview(
    *,
    range_key: str = "30",
    provider: str = "all",
    tool: str = "all",
) -> dict[str, Any]:
    global _last_result, _last_scan_at, _scans

    if range_key not in collectors.ALLOWED_RANGES:
        raise ValueError(f"unsupported range: {range_key}")
    if str(tool or "all").strip().casefold() not in collectors.ALLOWED_TOOLS:
        raise ValueError(f"unsupported tool: {tool}")

    started = time.monotonic()
    async with _scan_lock:
        data = await asyncio.to_thread(_scan, range_key, provider, tool)
        _last_result = data
        _last_scan_at = time.time()
        _scans += 1

    data["scan_seconds"] = round(time.monotonic() - started, 2)
    data["scans"] = _scans
    # 首次调用（缓存还是空的）耗时明显更长，前端据此提示"正在扫描本机轨迹"
    data["warming"] = data["scan_seconds"] >= 1.0
    return data


def status() -> dict[str, Any]:
    return {
        "scans": _scans,
        "last_scan_at": _last_scan_at or None,
        "has_result": _last_result is not None,
    }


async def prewarm() -> None:
    """启动后在后台预热一次，让第一个打开页面的人不必等冷扫描。

    只预热"全部工具 + 近 30 天"这一种组合：它是默认视图，也是缓存覆盖最广的一种
    （各数据源的解析缓存按文件/记录粒度复用，换 range 基本是纯内存聚合）。
    """
    try:
        await overview(range_key="30", provider="all", tool="all")
    except Exception:
        # 预热失败不能影响网关启动：缺数据源、库被占用等都属正常情况
        pass


def start_prewarm() -> None:
    """幂等启动预热任务；没有事件循环时静默跳过（例如在测试里同步调用）。"""
    global _prewarm_task
    if _prewarm_task is not None and not _prewarm_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _prewarm_task = loop.create_task(prewarm())
