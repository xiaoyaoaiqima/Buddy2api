#!/usr/bin/env bash
# 观察 tool-stall 误判是否还在发生 —— 直接查网关日志（最贴近真实使用）。
#
# 用法：
#   ./scripts/check_tool_stall.sh          # 只看今天的
#   ./scripts/check_tool_stall.sh 7        # 看最近 7 天
#
# 判读：
#   tool_stall 计数在修复后应保持不增长（除了真实停转）。
#   如果又出现，看 error_msg 里的正文：如果那是一段**完整的收尾回复**，
#   说明还有漏网的误判；如果是「好的，马上继续」这类确认话术，那是真停转、属正常修复行为。
set -euo pipefail
cd "$(dirname "$0")/.."
DAYS="${1:-1}"

echo "=== tool_stall 计数（最近 ${DAYS} 天）==="
sqlite3 -header -column codebuddy_gateway.db "
select date(created_at,'unixepoch','+8 hours') d,
       count(*) n,
       round(sum(prompt_tokens)/1000.0,1) || 'k' wasted_prompt_tokens
from logs
where finish_reason='tool_stall'
  and created_at >= strftime('%s','now') - ${DAYS}*86400
group by d order by d desc;"
echo
echo "（空结果 = 最近 ${DAYS} 天没有停转记录）"
echo
echo "=== 详情（若有）==="
sqlite3 -header -column codebuddy_gateway.db "
select datetime(created_at,'unixepoch','+8 hours') t,
       model, prompt_tokens, completion_tokens,
       substr(replace(coalesce(error_msg,''),char(10),' '),1,120) preview
from logs
where finish_reason='tool_stall'
  and created_at >= strftime('%s','now') - ${DAYS}*86400
order by created_at desc limit 20;"
