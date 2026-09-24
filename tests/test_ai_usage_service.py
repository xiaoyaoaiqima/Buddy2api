"""AI 用量接口层（`service`）的测试。

重点是**并发语义**，不是数值：冷扫描实测 ~25s，如果它跑在事件循环里，
那 25s 内所有转发请求都会停摆——这是能用测试抓住、而人眼很难发现的故障。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from buddy2api.ai_usage import collectors, service


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """每个用例都从干净状态开始，避免模块级计数互相干扰。"""
    monkeypatch.setattr(service, "_last_result", None)
    monkeypatch.setattr(service, "_last_scan_at", 0.0)
    monkeypatch.setattr(service, "_scans", 0)
    yield


def _fake_stats(range_key="30", provider="all", tool="all", **kwargs):
    # service 是按位置调用 _scan(range_key, provider, tool) 的
    return {
        "summary": {"tasks": 1, "tokens": 10, "requests": 1},
        "daily": [],
        "tools": [],
        "range_label": "近 30 日",
        "tool_label": "全部",
        "range": range_key,
        "provider": provider,
        "tool": tool,
    }


def test_rejects_unknown_range():
    with pytest.raises(ValueError):
        asyncio.run(service.overview(range_key="999"))


def test_rejects_unknown_tool():
    with pytest.raises(ValueError):
        asyncio.run(service.overview(tool="not-a-tool"))


def test_ok_result_is_not_marked_warming(monkeypatch):
    monkeypatch.setattr(service, "_scan", _fake_stats)
    data = asyncio.run(service.overview())
    assert data["warming"] is False
    assert data["scans"] == 1
    assert "scan_seconds" in data


def test_slow_result_is_marked_warming(monkeypatch):
    """冷扫描要能被前端识别出来，否则用户只能对着转圈猜是不是坏了。"""

    def slow(*args, **kwargs):
        time.sleep(1.2)
        return _fake_stats(*args, **kwargs)

    monkeypatch.setattr(service, "_scan", slow)
    data = asyncio.run(service.overview())
    assert data["warming"] is True
    assert data["scan_seconds"] >= 1.0


def test_scan_runs_off_the_event_loop(monkeypatch):
    """关键回归：扫描期间事件循环必须还能跑别的协程。

    这条测试如果用同步实现就会失败——把 build_ai_usage_stats 直接写进 async 路由
    正是要避免的做法。
    """
    ticks = 0

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return _fake_stats(*args, **kwargs)

    async def heartbeat():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.05)

    async def main():
        monkeypatch.setattr(service, "_scan", slow)
        beat = asyncio.create_task(heartbeat())
        await service.overview()
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass

    asyncio.run(main())
    assert ticks >= 5, f"扫描期间事件循环只跑了 {ticks} 次心跳，说明扫描阻塞了事件循环"


def test_concurrent_calls_do_not_scan_twice(monkeypatch):
    """连点刷新不该触发 N 次全盘扫描。"""
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        time.sleep(0.3)
        return _fake_stats(*args, **kwargs)

    async def main():
        monkeypatch.setattr(service, "_scan", counting)
        await asyncio.gather(*[service.overview() for _ in range(4)])

    asyncio.run(main())
    assert len(calls) == 4, "扫描被序列化了（锁生效），但每次都该真的扫一次"
    assert service.status()["scans"] == 4


def test_status_reports_last_scan(monkeypatch):
    monkeypatch.setattr(service, "_scan", _fake_stats)
    assert service.status()["has_result"] is False
    asyncio.run(service.overview())
    st = service.status()
    assert st["has_result"] is True
    assert st["scans"] == 1
    assert st["last_scan_at"] > 0


def test_prewarm_swallows_errors(monkeypatch):
    """预热失败不能影响网关启动：缺数据源、库被锁都属正常情况。"""

    def boom(*args, **kwargs):
        raise RuntimeError("数据源坏了")

    monkeypatch.setattr(service, "_scan", boom)
    asyncio.run(service.prewarm())  # 不该抛


def test_start_prewarm_is_safe_without_loop():
    """在同步上下文里调用不能炸（测试/脚本里会这么用）。"""
    service.start_prewarm()


def test_gateway_db_defaults_track_database_path(monkeypatch, tmp_path):
    """网关库路径必须跟 database.DB_PATH 同源，避免两处各写各的。"""
    target = tmp_path / "custom.db"
    monkeypatch.setenv("CB_GATEWAY_DB_PATH", str(target))
    assert collectors.default_gateway_db() == target


# ── 路由层 ──
# 移植自 zhiji 的 test_codex_usage_page_and_api：那条测的是 Flask 的
# /codex-usage + /api/codex-usage/overview，这里对应 buddy2api 的
# /admin/ai-usage/overview，顺带把鉴权也一起钉住。

@pytest.fixture
def _local_admin(monkeypatch):
    """把网关切到本机管理模式，路由才不需要 Admin Token。"""
    import buddy2api.server as server

    monkeypatch.setattr(server, "LOCAL_MODE", True)
    monkeypatch.setattr(server, "ALLOW_NO_ADMIN_AUTH", False)
    yield


def _route_request(path):
    import asyncio

    import httpx

    import buddy2api.server as server

    async def run():
        transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
            return await client.get(path)

    return asyncio.run(run())


def test_overview_route_requires_admin(_local_admin):
    """鉴权不能因为新增页面就漏掉：这是管理接口，会读出本机全部用量。"""
    import buddy2api.server as server

    server.LOCAL_MODE = False
    assert _route_request("/admin/ai-usage/overview").status_code == 401


def test_overview_route_passes_through_params(monkeypatch, _local_admin):
    seen = {}

    def fake(*args, **kwargs):
        # service 按位置调用 _scan(range_key, provider, tool)
        seen.update(zip(("range_key", "provider", "tool"), args))
        seen.update(kwargs)
        return _fake_stats(*args, **kwargs)

    monkeypatch.setattr(service, "_scan", fake)
    resp = _route_request("/admin/ai-usage/overview?range=7&tool=codex&provider=OpenAI")
    assert resp.status_code == 200
    assert resp.json()["summary"]["tasks"] == 1
    assert seen["range_key"] == "7", seen
    assert seen["tool"] == "codex", seen
    assert seen["provider"] == "OpenAI", seen


def test_overview_route_rejects_bad_range(monkeypatch, _local_admin):
    monkeypatch.setattr(service, "_scan", _fake_stats)
    resp = _route_request("/admin/ai-usage/overview?range=999")
    assert resp.status_code == 400


def test_status_route_reports_scan_count(monkeypatch, _local_admin):
    monkeypatch.setattr(service, "_scan", _fake_stats)
    resp = _route_request("/admin/ai-usage/status")
    assert resp.status_code == 200
    assert resp.json()["has_result"] is False
