"""AI 用量页的视觉契约测试。

这一页第一版做出来"能跑但难看"，问题都在**静态断言抓不到、但人一眼能看到**的地方。
把踩到的几条钉住，避免以后改动时无声退化。

这里刻意只断言"结构性的视觉缺陷"，不断言具体色值——配色可以调，
但"两种含义用同一个颜色""图例缺失""两栏高度差一倍"这类是缺陷，不是风格。
"""

import re
from pathlib import Path

import pytest

INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css(html) -> str:
    return html[html.find("<style>"): html.find("</style>")]


@pytest.fixture(scope="module")
def usage_src(html) -> str:
    i = html.find("component('usage'")
    j = html.find("component('stgs'")
    assert i > 0 and j > i
    return html[i:j]


def test_chart_is_tall_enough_to_read(css):
    """图表是这页的主体，不能挤成一条。

    第一版 190px，30 根柱每根 50px 宽，是"胖方块"而不是时序图。
    """
    m = re.search(r"\.uchart\{height:(\d+)px", css)
    assert m, "找不到 .uchart 高度"
    assert int(m.group(1)) >= 240, f"图表只有 {m.group(1)}px，太矮"


def test_chart_has_left_gutter_for_axis(css):
    """y 轴刻度是绝对定位的，柱区必须让出左边距，否则柱子压着数字。"""
    m = re.search(r"\.uchart-plot\{([^}]*)\}", css)
    assert m, "找不到 .uchart-plot"
    assert "padding-left" in m.group(1), "柱区没有给 y 轴留出左边距，柱子会压住刻度"


def test_chart_has_gridlines_and_y_axis(usage_src):
    """没有网格线和刻度，柱高就只能靠猜——这是"好看"之外的可用性问题。"""
    assert "ugrid" in usage_src, "缺少水平网格线"
    assert "uy-axis" in usage_src, "缺少 y 轴刻度"


def test_segment_colors_differ_by_hue_not_shade(css):
    """缓存输入与非缓存输入是图上最重要的对比，不能都用蓝色深浅表示。

    第一版是 #93b8f5 vs var(--blue)，两个蓝，肉眼几乎分不开。
    """
    cached = re.search(r"\.usepart-cached\{background:(#\w+|[^;}]+)", css)
    fresh = re.search(r"\.usepart-fresh\{background:(#\w+|[^;}]+)", css)
    assert cached and fresh, "缺少分层配色"

    def rgb(v):
        v = v.strip()
        if v.startswith("#") and len(v) == 7:
            return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
        return None

    h1, h2 = rgb(cached.group(1)), rgb(fresh.group(1))
    if h1 and h2:
        # 色相差异：最大通道与最小通道的顺序必须不同，否则是同一色系的深浅
        def order(c):
            return tuple(sorted(range(3), key=lambda i: c[i]))
        assert order(h1) != order(h2), (
            f"缓存({cached.group(1)}) 与非缓存({fresh.group(1)}) 是同一色相，图上分不开"
        )


def test_legend_exists_and_is_dynamic(usage_src):
    """堆叠图里颜色是有含义的，没有图例等于四种噪声。"""
    assert "ulegend" in usage_src, "图表没有图例"
    assert "legendOf" in usage_src, "图例没有随度量切换（看输出 Token 时还显示缓存层就是错的）"


def test_tooltip_is_a_real_element_not_native_title(usage_src, css):
    """原生 title 延迟约 1 秒、不能排版。数据这么密的一页要用自绘浮层。"""
    assert 'class="utip"' in usage_src, "没有自绘浮层"
    assert "tipRows" in usage_src, "浮层没有内容函数"
    assert "tipStyle" in usage_src, "浮层没有贴边翻转（会被 overflow 裁掉）"
    assert ".utip{" in css, "浮层没有样式"
    # 悬浮提示不该再依赖原生 title
    bar = re.search(r"<button[^>]*class=\"ubar-wrap\"[^>]*>", usage_src)
    assert bar, "找不到柱子按钮"
    assert ":title=" not in bar.group(0), "柱子又用回了原生 title"


def test_bars_have_hover_feedback(css):
    assert ".ubar-wrap:hover" in css, "柱子没有 hover 反馈"


def test_editorial_layout_avoids_the_card_grid(usage_src):
    """不要"内容切成 N 张同样的圆角卡片"。

    frontend-design skill 把这条点名为生成感第 4 特征原文：
    "the SaaS-card kit: content chopped into identical rounded cards"。
    上一版有 11 张同款卡片，是本页"模板感"的根源。改成细分隔线分区后应为 0。
    """
    cards = len(re.findall(r'class="card"', usage_src))
    assert cards == 0, f"这一页又用回了 {cards} 张卡片式布局"
    assert "ublock" in usage_src, "没有无边框分区（.ublock）"


def test_main_number_leads_the_page(usage_src, css):
    """主数必须是页面上最大的文字，且远大于次级标签。

    用字号跳跃建立层级（44 → 20 → 12），而不是靠给每块加个框。
    """
    hero = re.search(r"\.uhero-num\{([^}]*)\}", css)
    assert hero, "没有主数样式"
    size = int(re.search(r"font-size:(\d+)px", hero.group(1)).group(1))
    assert size >= 34, f"主数只有 {size}px，主体不突出"
    fact = re.search(r"\.ufact-n\{([^}]*)\}", css)
    assert fact, "没有次级数字样式"
    sub = int(re.search(r"font-size:(\d+)px", fact.group(1)).group(1))
    assert size >= sub * 1.8, f"主数({size}) 与次级数字({sub}) 差距不足，层级不清"


def test_insight_facts_are_named_not_decorative(usage_src):
    """辅助数字必须自带说明，否则读者不知道 79% 是什么的 79%。

    每张"事实"都由数字 + 一句解释组成，禁止裸数字堆砌。
    """
    # 每个 ufact 是单行，用非贪婪匹配到自己那个 </div> 结束；
    # 写成 `</div>\s*</div>` 会跨块吞并，导致只匹配到 1 个。
    facts = re.findall(r'<div class="ufact">.*?</div>', usage_src, re.S)
    assert len(facts) >= 2, "辅助事实块不足"
    for f in facts:
        assert "ufact-n" in f, "事实块缺少数字"
        assert "ufact-l" in f, "事实块缺少解释文字（裸数字读者无法理解）"


def test_no_midpoint_meta_strings(usage_src):
    """skill 点名的 'A · B · C' 中点串是模板化特征之一。

    表头与标签用空格或分隔线，不用中点串堆信息。
    """
    # 允许"用 · 连接两个并列项"的极少数情况（如工具 · 项目），但不能出现在标题/说明里
    h3 = re.findall(r"<h3>(.*?)</h3>", usage_src, re.S)
    offenders = [h for h in h3 if "·" in h]
    assert not offenders, f"分区标题里用了中点串：{offenders}"


def test_recent_tasks_prioritize_nonzero_tokens(usage_src):
    """最近任务里近一半是 0 Token 的会话，按时间混排会把有信息量的挤下去。"""
    assert "recentSorted" in usage_src, "最近任务没有做排序"
    assert "utask.zero" in usage_src or "zero:!t.tokens" in usage_src, "0 Token 行没有弱化"


# ── 可读性：对比度与字号 ──
# 这两条来自一次真实审计：整页小字用了 --fg3(#999)，在白底上只有 2.85:1，
# 低于 WCAG AA 对小字的 4.5:1；而且有 55 处字号 <11px。截图里"看不清"就是这么来的。

def _luminance(color: str):
    color = color.strip()
    if color.startswith("#"):
        if len(color) != 7:
            return None
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    else:
        nums = re.findall(r"\d+", color)
        if len(nums) < 3:
            return None
        rgb = [int(v) / 255 for v in nums[:3]]

    def f(v):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (f(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg: str, bg: str = "#ffffff"):
    a, b = _luminance(fg), _luminance(bg)
    if a is None or b is None:
        return None
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def test_data_text_meets_wcag_aa(css):
    """承载数据的小字必须达到 AA 4.5:1。

    只查这一页自己的类，不去动全站共享的 --fg3 —— 那是既有问题，
    不该由一个页面的改动单方面改掉全局 token。
    """
    # 这些选择器承载的是数字/标签本身，不是装饰
    data_text = [".ukpi-label", ".ukpi-hint", ".uchart-stat-label", ".urank-sub",
                 ".utask-meta", ".uinsight-label", ".ufootnote", ".ux-axis", ".uy-axis"]
    offenders = []
    for sel in data_text:
        m = re.search(re.escape(sel) + r"\{([^}]*)\}", css)
        if not m:
            continue
        body = m.group(1)
        if "var(--fg3)" not in body:
            continue
        offenders.append(sel)
    assert not offenders, (
        f"这些承载数据的小字用了 --fg3（白底 2.85:1，低于 AA 4.5:1）：{offenders}"
    )


def test_fg3_really_is_too_low_contrast():
    """守住上面那条断言的前提：--fg3 确实不达标，--fg2 达标。

    如果哪天有人调亮了 --fg3，这条会失败并提醒：上面的规则可以放宽了。
    """
    assert contrast("#999999") < 4.5, "#999 竟然达标了，规则需要复核"
    assert contrast("#666666") >= 4.5, "--fg2 不达标，数据小字没有安全的去处"


def test_no_microscopic_text(css):
    """这一页自己的样式里不该有 9-10px 的正文文本。

    审计时全页 55 处 <11px 是"看不清"的主因。**只查本页新增的 `u` 前缀类**：
    全站既有的小字号是别处的历史问题，不该借一个页面的改动去判定。

    例外：坐标轴刻度用 10px 等宽字是图表惯例（密度高、单字符、不承载句子），
    明确列在白名单里，避免这条规则退化成"不许有例外"的死规矩。
    """
    allowed_tiny = {"uy-axis", "ux-axis"}
    bad = []
    for m in re.finditer(r"\.(u[a-z0-9-]+)\{([^}]*)\}", css):
        cls, body = m.group(1), m.group(2)
        fs = re.search(r"font-size:(\d+)px", body)
        if fs and int(fs.group(1)) < 11 and cls not in allowed_tiny:
            bad.append(f".{cls} {fs.group(0)}")
    assert not bad, f"本页仍有小于 11px 的正文样式：{bad}"


def test_headline_content_outweighs_the_tail(css, usage_src):
    """图表高度不能低于任务列表——信息层级不能倒置。

    审计时图表 260px、任务列表 1119px（20 条），主视觉被列表压过去。
    """
    m = re.search(r"\.uchart\{height:(\d+)px", css)
    assert m and int(m.group(1)) >= 280, "图表又被压矮了"
    cap = re.search(r"recentSorted\.slice\(0,(\d+)\)", usage_src)
    assert cap, "最近任务没有限制条数，会把图表比例压垮"
    assert int(cap.group(1)) <= 10, f"最近任务渲染 {cap.group(1)} 条，太多"


# ── 减法：删掉的块不许悄悄回来 ──
# 依据是实测的冗余度（见 commit message）：
#   「值得注意」4 条里 2 条与别处完全重复（30/30 活跃日恒为满勤；网关 61% 已在顶部）
#   模型表 35 个模型前 7 占 95%，长尾 28 个占 5% —— 压成一句话脚注
# 删掉比留着好，但以后可能有人"顺手加回来"，所以钉住。

def test_no_duplicate_insight_block(usage_src):
    """「值得注意」整块已删：其中 2/4 条与别处重复，属于自我重复的噪声。"""
    assert "unotes" not in usage_src, "「值得注意」块又被加回来了"
    assert "d.insights" not in usage_src, "又在渲染 insights 列表"


def test_model_list_is_summarized_not_enumerated(usage_src):
    """模型维度收成一句话，不再列 7 行（前 7 占 95%，长尾无决策价值）。"""
    assert "d.models.slice(0,7)" not in usage_src, "模型表又变回 7 行长表"
    assert "topModel" in usage_src, "模型维度没有收口成摘要"
    assert "ublock-note" in usage_src, "缺少长尾收口说明"


def test_recent_sessions_keep_project_column(usage_src):
    """最近会话必须保留「项目」列。

    我一度把它换成「工具」——那是减过头：工具分布已由「构成」表达，
    而"这次会话属于哪个项目"只有这里能看到。
    """
    recent = re.search(r"<h3>最近会话.*?</table>", usage_src, re.S)
    assert recent, "找不到最近会话块"
    body = recent.group(0)
    assert "t.project" in body, "最近会话丢掉了项目列"
    assert "t.tool" not in body, "最近会话又用工具列占位（工具已由「构成」表达）"


def test_block_names_do_not_collide(usage_src):
    """区块名不能和页底的口径说明撞名（都叫"注脚"读起来像同一件事）。"""
    heads = re.findall(r"<h3>([^<]*)", usage_src)
    assert not any("注脚" in h for h in heads), f"区块名里还有'注脚'：{heads}"


def test_hero_facts_each_express_one_thing(usage_src):
    """顶部每个数字只讲一件事，不在解释里再塞三组计数。"""
    facts = re.findall(r'<div class="ufact">.*?</div>', usage_src, re.S)
    for f in facts:
        label = re.search(r'ufact-l">(.*?)</span>', f, re.S)
        if not label:
            continue
        text = label.group(1)
        # 解释里不该再堆多个 {{n(...)}} 计数
        assert text.count("{{n(") <= 1, f"事实说明里塞了多个计数：{text}"
