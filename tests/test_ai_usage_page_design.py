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


def test_two_columns_are_roughly_balanced(usage_src):
    """两栏高度差一倍会让短的那栏拖出大片空白。

    第一版左栏 361px / 右栏 637px。做法上把「模型分布」从独占一行挪进左栏。
    """
    grid = re.search(r'<div class="ugrid2">(.*?)\n    </div>', usage_src, re.S)
    assert grid, "找不到两栏容器"
    body = grid.group(1)
    # 左栏应该是"若干张短卡片"的容器，右栏是被限制行数的长列表
    assert body.count('class="card"') >= 3, "两栏里应该有 3 张卡片（工具/模型 + 项目）"
    assert "slice(0,12)" in body or "slice(0,10)" in body, (
        "长的那栏没有限制行数，会把另一栏拖出巨大空白"
    )


def test_recent_tasks_prioritize_nonzero_tokens(usage_src):
    """最近任务里近一半是 0 Token 的会话，按时间混排会把有信息量的挤下去。"""
    assert "recentSorted" in usage_src, "最近任务没有做排序"
    assert "utask.zero" in usage_src or "zero:!t.tokens" in usage_src, "0 Token 行没有弱化"
