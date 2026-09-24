import json
import sqlite3
from datetime import datetime
from pathlib import Path

from buddy2api.ai_usage.codex_usage import _parse_rollout, _read_rollout, build_codex_usage_stats


def _counter(timestamp, total, last=None):
    info = {"total_token_usage": {"input_tokens": total, "total_tokens": total}}
    if last is not None:
        info["last_token_usage"] = {"input_tokens": last, "total_tokens": last}
    return {"timestamp": timestamp, "type": "event_msg",
            "payload": {"type": "token_count", "info": info}}


def test_rotated_rollouts_keep_history_on_original_dates_and_invalidate_cache(tmp_path, monkeypatch):
    from buddy2api.ai_usage import codex_usage as stats

    state = tmp_path / "state.sqlite"
    _create_state_db(state)
    thread_id = "01a08dfd-ed52-7f30-b561-a1e8a401db57"
    old_dir = tmp_path / "archived_sessions"
    new_dir = tmp_path / "sessions" / "2026" / "09" / "13"
    old_dir.mkdir()
    new_dir.mkdir(parents=True)
    old = old_dir / f"rollout-2026-09-12T23-00-00-{thread_id}.jsonl"
    new = new_dir / f"rollout-2026-09-13T00-18-51-{thread_id}_rotation.jsonl"
    previous = _counter("2026-09-12T15:00:00Z", 110_000_000)
    today = _counter("2026-09-12T16:19:00Z", 110_050_000, 50_000)
    old.write_text(json.dumps(previous) + "\n")
    # A copied event at the shard boundary must not count a second time.
    new.write_text(json.dumps(previous) + "\n" + json.dumps(today) + "\n")
    with sqlite3.connect(state) as conn:
        conn.execute("UPDATE threads SET id=?,rollout_path=? WHERE id='root-1'",
                     (thread_id, str(new)))
    cache = tmp_path / "usage.sqlite"
    kwargs = dict(state_db=state, cache_db=cache, ttl_seconds=0)
    stats._refresh_usage_cache(**kwargs)
    with sqlite3.connect(cache) as conn:
        rows = dict(conn.execute("SELECT usage_date,total_tokens FROM daily_usage WHERE thread_id=?", (thread_id,)))
    assert rows == {"2026-09-12": 110_000_000, "2026-09-13": 50_000}
    # An old shard changes but the DB pointer/current shard does not.
    old.write_text(json.dumps(previous) + "\n" + json.dumps(_counter("2026-09-12T15:30:00Z", 110_020_000)) + "\n")
    stats._refresh_usage_cache(**kwargs)
    with sqlite3.connect(cache) as conn:
        rows = dict(conn.execute("SELECT usage_date,total_tokens FROM daily_usage WHERE thread_id=?", (thread_id,)))
    assert rows == {"2026-09-12": 110_020_000, "2026-09-13": 30_000}
    def unexpected_read(_paths):
        raise AssertionError("unchanged shards should use persistent cache")
    monkeypatch.setattr(stats, "_read_rollout_shards", unexpected_read)
    stats._refresh_usage_cache(**kwargs)


def test_rotated_fragment_does_not_assign_unknown_history_to_today(tmp_path):
    from buddy2api.ai_usage.codex_usage import _read_rollout_shards, _summarize_rollout_events

    path = tmp_path / "rollout-2026-09-13T00-18-51-session_rotation.jsonl"
    path.write_text(json.dumps(_counter("2026-09-12T16:19:00Z", 110_050_000, 50_000)) + "\n")
    _, events = _read_rollout_shards([path])
    daily = _summarize_rollout_events(events)
    assert daily["2026-09-13"]["total_tokens"] == 50_000


def _create_state_db(path: Path) -> None:
    rollout_paths = {}
    events = {
        "root-1": [
            ("2026-08-10T02:00:00Z", 90, 60, 10, 100),
            ("2026-08-10T02:01:00Z", 90, 60, 10, 100),
            ("2026-08-11T02:00:00Z", 140, 100, 20, 160),
            ("2026-08-11T02:01:00Z", 135, 95, 20, 155),
            ("2026-08-11T02:02:00Z", 145, 103, 20, 165),
            ("2026-08-12T02:00:00Z", 15, 5, 5, 20),
        ],
        "root-2": [("2026-08-12T04:00:00Z", 180, 120, 20, 200)],
        "child-1": [("2026-08-12T05:00:00Z", 45, 30, 5, 50)],
    }
    for thread_id, thread_events in events.items():
        rollout_path = path.parent / f"{thread_id}.jsonl"
        rollout_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": input_tokens,
                                    "cached_input_tokens": cached_tokens,
                                    "cache_write_input_tokens": 0,
                                    "output_tokens": output_tokens,
                                    "reasoning_output_tokens": 0,
                                    "total_tokens": total_tokens,
                                }
                            },
                        },
                    }
                )
                for timestamp, input_tokens, cached_tokens, output_tokens, total_tokens in thread_events
            )
            + "\n",
            encoding="utf-8",
        )
        rollout_paths[thread_id] = str(rollout_path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                rollout_path TEXT,
                title TEXT,
                first_user_message TEXT,
                cwd TEXT,
                model TEXT,
                source TEXT,
                tokens_used INTEGER,
                created_at INTEGER,
                updated_at INTEGER,
                archived INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "root-1",
                    rollout_paths["root-1"],
                    "zhiji task",
                    "",
                    "/Users/test/zhiji",
                    "gpt-test",
                    "vscode",
                    1000,
                    int(datetime(2026, 8, 10, 10).timestamp()),
                    int(datetime(2026, 8, 10, 11).timestamp()),
                    0,
                ),
                (
                    "root-2",
                    rollout_paths["root-2"],
                    "raap task",
                    "",
                    "/Users/test/rs_raap",
                    "gpt-test",
                    "cli",
                    2000,
                    int(datetime(2026, 8, 12, 10).timestamp()),
                    int(datetime(2026, 8, 12, 12).timestamp()),
                    1,
                ),
                (
                    "child-1",
                    rollout_paths["child-1"],
                    "sub task",
                    "",
                    "/Users/test/zhiji",
                    "gpt-test",
                    '{"subagent":{"other":"guardian"}}',
                    500,
                    int(datetime(2026, 8, 12, 10).timestamp()),
                    int(datetime(2026, 8, 12, 11).timestamp()),
                    0,
                ),
            ],
        )


def test_build_codex_usage_stats_separates_root_and_subagent_tasks(tmp_path):
    db_path = tmp_path / "state.sqlite"
    _create_state_db(db_path)

    result = build_codex_usage_stats(
        state_db=db_path,
        cache_db=tmp_path / "usage-cache.sqlite",
        range_key="7",
        now=datetime(2026, 8, 13, 9),
        cache_ttl_seconds=0,
    )

    assert result["summary"]["tasks"] == 2
    assert result["summary"]["tokens"] == 435
    assert result["summary"]["input_tokens"] == 385
    assert result["summary"]["cached_input_tokens"] == 258
    assert result["summary"]["output_tokens"] == 50
    assert result["summary"]["involved_threads"] == 3
    assert result["summary"]["subagent_tasks"] == 1
    assert result["summary"]["root_tokens"] == 385
    assert result["summary"]["subagent_tokens"] == 50
    assert result["summary"]["active_days"] == 3
    assert len(result["daily"]) == 7
    assert result["daily"][1]["date"] == "2026-08-08"
    assert result["daily"][1]["tasks"] == 0
    assert result["daily"][1]["tokens"] == 0
    assert result["daily"][3]["tasks"] == 1
    assert result["daily"][3]["tokens"] == 100
    assert result["daily"][5]["tokens"] == 270
    assert result["daily"][5]["involved_threads"] == 3
    assert result["daily"][5]["root_tokens"] == 220
    assert result["daily"][5]["subagent_tokens"] == 50
    assert result["daily"][5]["tokens_per_thread"] == 90
    assert result["daily"][5]["cache_ratio"] == 0.6458
    assert result["projects"][0]["tasks"] == 1
    with sqlite3.connect(tmp_path / "usage-cache.sqlite") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_build_codex_usage_stats_rejects_unknown_range(tmp_path):
    db_path = tmp_path / "state.sqlite"
    _create_state_db(db_path)

    try:
        build_codex_usage_stats(state_db=db_path, range_key="14")
    except ValueError as exc:
        assert "unsupported range" in str(exc)
    else:
        raise AssertionError("unknown range should fail")


def test_parse_rollout_counts_only_child_delta_after_copied_parent_history(tmp_path):
    def token_event(timestamp, input_tokens, cached_tokens, output_tokens, total_tokens, last):
        return {
            "timestamp": timestamp,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_tokens,
                        "cache_write_input_tokens": 0,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": 0,
                        "total_tokens": total_tokens,
                    },
                    "last_token_usage": last,
                },
            },
        }

    first = token_event(
        "2026-08-10T02:00:00Z",
        100,
        70,
        10,
        110,
        {
            "input_tokens": 100,
            "cached_input_tokens": 70,
            "cache_write_input_tokens": 0,
            "output_tokens": 10,
            "reasoning_output_tokens": 0,
            "total_tokens": 110,
        },
    )
    second = token_event(
        "2026-08-10T02:01:00Z",
        150,
        105,
        15,
        165,
        {
            "input_tokens": 50,
            "cached_input_tokens": 35,
            "cache_write_input_tokens": 0,
            "output_tokens": 5,
            "reasoning_output_tokens": 0,
            "total_tokens": 55,
        },
    )
    child_delta = token_event(
        "2026-08-12T02:00:00Z",
        180,
        126,
        18,
        198,
        {
            "input_tokens": 30,
            "cached_input_tokens": 21,
            "cache_write_input_tokens": 0,
            "output_tokens": 3,
            "reasoning_output_tokens": 0,
            "total_tokens": 33,
        },
    )
    # Older parent rollouts can omit fields whose value is zero, while a
    # forked child writes those same fields explicitly.  They are still the
    # same inherited token history and must be de-duplicated.
    parent_first = json.loads(json.dumps(first))
    parent_second = json.loads(json.dumps(second))
    for item in (parent_first, parent_second):
        for usage_key in ("total_token_usage", "last_token_usage"):
            item["payload"]["info"][usage_key].pop("cache_write_input_tokens")

    parent_path = tmp_path / "parent.jsonl"
    parent_path.write_text(
        "\n".join(json.dumps(item) for item in [parent_first, parent_second]) + "\n",
        encoding="utf-8",
    )
    child_path = tmp_path / "child.jsonl"
    child_path.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {"type": "session_meta", "payload": {"forked_from_id": "parent"}},
                first,
                second,
                child_delta,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, parent_events = _read_rollout(parent_path)
    result = _parse_rollout(child_path, inherited_events=parent_events)

    assert result == {
        "2026-08-12": {
            "input_tokens": 30,
            "cached_input_tokens": 21,
            "cache_write_input_tokens": 0,
            "output_tokens": 3,
            "reasoning_output_tokens": 0,
            "total_tokens": 33,
        }
    }


def test_parse_rollout_uses_last_usage_for_legacy_fork_counter(tmp_path):
    parent_path = tmp_path / "parent.jsonl"
    parent_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-10T02:00:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"total_tokens": 165}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child_path = tmp_path / "legacy-child.jsonl"
    child_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-12T02:00:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {"total_tokens": 198},
                        "last_token_usage": {"total_tokens": 33},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, parent_events = _read_rollout(parent_path)
    result = _parse_rollout(child_path, inherited_events=parent_events)

    assert result["2026-08-12"]["total_tokens"] == 33


def test_build_codex_usage_stats_groups_root_tasks_by_provider(tmp_path):
    db_path = tmp_path / "state.sqlite"
    _create_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE threads ADD COLUMN model_provider TEXT")
        conn.execute("UPDATE threads SET model_provider = 'OpenAI' WHERE id = 'root-1'")
        conn.execute("UPDATE threads SET model_provider = 'cliproxy' WHERE id = 'root-2'")
        conn.execute("UPDATE threads SET model_provider = 'cliproxy' WHERE id = 'child-1'")
    result = build_codex_usage_stats(
        state_db=db_path,
        cache_db=tmp_path / "usage-cache.sqlite",
        range_key="7",
        now=datetime(2026, 8, 13, 9),
        cache_ttl_seconds=0,
    )

    assert result["providers"] == [
        {"name": "cliproxy", "tasks": 1, "tokens": 200},
        {"name": "OpenAI", "tasks": 1, "tokens": 185},
    ]
    assert result["recent_tasks"][0]["provider"] == "cliproxy"


def test_build_codex_usage_stats_merges_historical_provider_aliases(tmp_path):
    db_path = tmp_path / "state.sqlite"
    _create_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE threads ADD COLUMN model_provider TEXT")
        conn.execute("UPDATE threads SET model_provider = 'goumanbuka' WHERE id = 'root-1'")
        conn.execute("UPDATE threads SET model_provider = 'raap' WHERE id = 'root-2'")
        conn.execute("UPDATE threads SET model_provider = 'raap' WHERE id = 'child-1'")

    result = build_codex_usage_stats(
        state_db=db_path,
        cache_db=tmp_path / "usage-cache.sqlite",
        range_key="7",
        now=datetime(2026, 8, 13, 9),
        cache_ttl_seconds=0,
    )

    assert result["providers"] == [
        {"name": "cliproxy", "tasks": 1, "tokens": 200},
        {"name": "custom", "tasks": 1, "tokens": 185},
    ]
    assert result["recent_tasks"][0]["provider"] == "cliproxy"

    filtered = build_codex_usage_stats(
        state_db=db_path,
        cache_db=tmp_path / "usage-cache.sqlite",
        range_key="7",
        provider="raap",
        now=datetime(2026, 8, 13, 9),
        cache_ttl_seconds=0,
    )

    assert filtered["provider"] == "cliproxy"
    assert filtered["summary"]["tasks"] == 1
    assert filtered["summary"]["tokens"] == 250


def test_build_codex_usage_stats_filters_projects_and_tokens_by_provider(tmp_path):
    db_path = tmp_path / "state.sqlite"
    _create_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE threads ADD COLUMN model_provider TEXT")
        conn.execute("UPDATE threads SET model_provider = 'OpenAI' WHERE id = 'root-1'")
        conn.execute("UPDATE threads SET model_provider = 'cliproxy' WHERE id = 'root-2'")
        conn.execute("UPDATE threads SET model_provider = 'cliproxy' WHERE id = 'child-1'")

    result = build_codex_usage_stats(
        state_db=db_path,
        cache_db=tmp_path / "usage-cache.sqlite",
        range_key="7",
        provider="cliproxy",
        now=datetime(2026, 8, 13, 9),
        cache_ttl_seconds=0,
    )

    assert result["provider"] == "cliproxy"
    assert result["provider_label"] == "cliproxy"
    assert result["summary"]["tasks"] == 1
    assert result["summary"]["tokens"] == 250
    assert result["summary"]["involved_threads"] == 2
    assert result["summary"]["subagent_tasks"] == 1
    assert result["summary"]["root_tokens"] == 200
    assert result["summary"]["subagent_tokens"] == 50
    assert result["projects"] == [
        {"name": "rs_raap", "cwd": "/Users/test/rs_raap", "tasks": 1, "tokens": 200}
    ]
    assert result["project_daily"]["2026-08-12"] == [
        {"name": "rs_raap", "cwd": "/Users/test/rs_raap", "tasks": 1, "tokens": 200},
        {"name": "zhiji", "cwd": "/Users/test/zhiji", "tasks": 1, "tokens": 50},
    ]
    assert result["daily"][5]["tokens"] == 250
    assert result["daily"][5]["root_tokens"] == 200
    assert result["daily"][5]["subagent_tokens"] == 50
    assert result["daily"][5]["tokens_per_thread"] == 125
    assert result["recent_tasks"][0]["provider"] == "cliproxy"
    assert result["providers"] == [
        {"name": "cliproxy", "tasks": 1, "tokens": 200},
        {"name": "OpenAI", "tasks": 1, "tokens": 185},
    ]
