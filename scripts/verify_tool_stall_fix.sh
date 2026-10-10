#!/usr/bin/env bash
# 验证 tool-stall 误判修复是否在运行中的网关上生效。
#
# 原理：构造一个「agent 工具循环回合」（带 tools + 历史含 role=tool），
# 让模型输出一段**不含任何正向话术的短收尾回复**。修复前这种回复会被判成
# 停转并触发 tool_choice=required 重打；修复后应当原样返回。
#
# 判据（无需读日志）：
#   - 返回 finish_reason=stop 且正文是收尾语  → 修复生效（没有重打、没有报错）
#   - 返回 error.code=upstream_tool_stall      → 修复未生效
#   - 返回 finish_reason=tool_calls            → 修复未生效（被逼着调工具）
#
# 用法：./scripts/verify_tool_stall_fix.sh [模型名]
set -euo pipefail

BASE="${CB_GATEWAY_BASE:-http://127.0.0.1:8787}"
MODEL="${1:-deepseek-v4.1-flash}"

KEY="${BUDDY2API_KEY:-}"
if [[ -z "$KEY" ]]; then
  KEY=$(python3 - <<'PY'
import re, pathlib
p = pathlib.Path.home() / ".dsh/.credentials.yaml"
m = re.search(r"BUDDY2API_KEY:\s*(\S+)", p.read_text())
print(m.group(1) if m else "")
PY
)
fi
if [[ -z "$KEY" ]]; then
  echo "找不到 BUDDY2API_KEY；请设置环境变量或用 --key 传入。" >&2
  exit 2
fi

# 收尾回复：短、明确表示「做完了/在等你」，但不含任何 _STALL_POSITIVE_MARKERS。
# 修复前 100% 被判停转 —— 这正是生产日志里误伤的那类文本。
read -r -d '' BODY <<'JSON' || true
{
  "model": "__MODEL__",
  "stream": false,
  "messages": [
    {"role": "user", "content": "继续跑流程。"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "bash", "arguments": "{\"command\":\"echo ok\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    {"role": "user", "content": "不用再动手了。用一两句话告诉我刚才那条命令的结果是什么，直接回答，不要调用任何工具。"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "bash",
      "description": "run a shell command",
      "parameters": {"type": "object",
                     "properties": {"command": {"type": "string"}},
                     "required": ["command"]}}}
  ],
  "max_tokens": 256
}
JSON

BODY="${BODY/__MODEL__/$MODEL}"

echo "→ 网关: $BASE"
echo "→ 模型: $MODEL"
echo "→ 期望: finish_reason=stop，正文为收尾语（不重打、不报 upstream_tool_stall）"
echo

RESP=$(curl -sS --max-time 180 "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY")

echo "$RESP" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("非 JSON 响应："); print(raw[:800]); sys.exit(2)

if "error" in d and d["error"]:
    err = d["error"]
    print("❌ 失败:", err.get("code"), "-", str(err.get("message"))[:200])
    if err.get("code") == "upstream_tool_stall":
        print("   → 这正是被修复的误判：修复未生效（网关仍在跑旧逻辑）。")
    sys.exit(1)

ch = (d.get("choices") or [{}])[0]
msg = ch.get("message") or {}
fr = ch.get("finish_reason")
tc = msg.get("tool_calls")
content = (msg.get("content") or "").strip()

print("finish_reason :", fr)
print("tool_calls    :", "有" if tc else "无")
print("content       :", content[:400].replace("\n", " ⏎ "))
print()

if fr == "stop" and not tc and content:
    print("✅ 修复生效：收尾回复原样返回，没有被判停转、没有重打。")
    sys.exit(0)
if tc:
    print("❌ 修复未生效：模型被逼着调工具（tool_choice=required 重打路径仍在触发）。")
    sys.exit(1)
print("⚠️  结果不符合任一预期，请人工判断。")
sys.exit(3)
'
