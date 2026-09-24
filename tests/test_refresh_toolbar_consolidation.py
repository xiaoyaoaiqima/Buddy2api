"""工具栏刷新按钮合并的契约测试。

背景：账号页原先并排三个刷新按钮 ——「刷新列表」「刷新官方额度」「刷新领取状态」。
其中后两个是第一个的真子集：`load(true)` 内部就是
    取列表 → loadCheckins() → refreshAllResources()
而且「刷新官方额度」调 `refreshAllResources()`（force=false），与 `load(true)` 的
第三步发的是**完全相同的请求**，点哪个都一样。

**而且连「强制刷新」都不需要**：签到缓存是当天维度的（checkin_date + today_only），
签到本身又是每天一次的动作，所以"状态变了但缓存旧"的窗口只剩「你在网页外刚领完」
这一种，5 分钟自愈。真正需要强制重读的两处（claimOne / claimAll）内部本来就调
loadCheckins(true)，不依赖按钮。所以最终只留一个「刷新」。

这里钉死：三个旧按钮没了、只剩一个「刷新」、它就是全量刷新（列表+领取状态+官方额度），
且领取后仍会强制重读签到状态。
"""

import re
from pathlib import Path

import pytest

INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "index.html"


@pytest.fixture(scope="module")
def html_source() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def account_toolbar(html_source) -> str:
    """账号页工具栏那一行。"""
    for line in html_source.split("\n"):
        if "claimAll" in line and "tbar" in line:
            return line
    pytest.fail("找不到账号页工具栏")


def test_redundant_buttons_removed(account_toolbar):
    """三个旧按钮的文字都不应再出现在工具栏里。"""
    for label in ("刷新列表", "刷新官方额度", "刷新领取状态"):
        assert label not in account_toolbar, f"工具栏里仍有冗余按钮「{label}」"


def test_single_refresh_button_does_everything(account_toolbar):
    """保留的「刷新」必须还是 load(true)（列表 + 领取状态 + 官方额度）。"""
    assert '@click="load(true)"' in account_toolbar, "「刷新」不再是全量刷新"
    assert ">刷新<" in account_toolbar.replace("'刷新中':'刷新'", ">刷新<") or "'刷新'" in account_toolbar


def test_only_one_refresh_button_remains(account_toolbar):
    """工具栏里只该有一个刷新按钮——多一个都是让用户犹豫的设计。"""
    refresh_btns = re.findall(r'<button[^>]*>(?:(?!</button>).)*?刷新(?:(?!</button>).)*?</button>', account_toolbar)
    assert len(refresh_btns) == 1, (
        f"工具栏里有 {len(refresh_btns)} 个刷新按钮，应该只剩 1 个：{[re.sub(chr(60) + '[^>]+' + chr(62), '', b) for b in refresh_btns]}"
    )
    assert "强制刷新" not in account_toolbar, "「强制刷新」又回来了"


def test_load_no_longer_takes_force(html_source):
    """load 回到单参数：没有调用方需要绕过缓存了，留着参数=留着一个没人用的旋钮。"""
    m = re.search(r"async function load\(withOfficial=false[^)]*\)", html_source)
    assert m, "load 的签名变了"
    assert "force" not in m.group(0), f"load 还带着没人用的 force 参数：{m.group(0)}"


def test_claim_still_forces_checkin_reread(html_source):
    """领完必须强制重读签到状态，这条能力不能因为删按钮而丢掉。"""
    for fn in ("claimOne", "claimAll"):
        # 这两个函数整体写在一行，不能用 \n 收尾。
        i = html_source.find(f"async function {fn}(")
        assert i > 0, f"找不到 {fn}"
        body = html_source[i : html_source.find("\n", i)]
        assert "loadCheckins(true)" in body, (
            f"{fn} 领完之后没有强制重读签到状态，会显示旧的「可领取」"
        )


def test_refresh_button_has_title(account_toolbar):
    """一个按钮也要说清它刷了什么，否则用户不知道「刷新」的范围。"""
    assert "title=" in account_toolbar
    assert "官方额度" in account_toolbar and "领取" in account_toolbar


def test_claim_and_seamless_buttons_untouched(account_toolbar):
    """这次只删刷新类按钮，别把无关功能删掉。"""
    for label in ("一键领取今日积分", "无感登录", "高级手动添加", "重置请求计数"):
        assert label in account_toolbar, f"误删了「{label}」"
