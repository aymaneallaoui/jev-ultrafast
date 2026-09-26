import json
import shutil
import subprocess

import pytest

from jev_ultrafast import browser


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def cdp(method, **params):
        calls.append((method, params))
        return {
            "Target.createTarget": {"targetId": "target"},
            "Target.attachToTarget": {"sessionId": "session"},
            "Runtime.evaluate": {"result": {"value": "complete"}},
        }.get(method, {})

    monkeypatch.delenv("JEV_OWN_WINDOW", raising=False)
    monkeypatch.delenv("JEV_VIEWPORT_WIDTH", raising=False)
    monkeypatch.setattr(browser, "cdp", cdp)
    monkeypatch.setattr(browser, "ensure_daemon", lambda: None)
    return calls


def params(calls, method):
    return next(p for m, p in calls if m == method)


def test_default_opens_background_tab_at_1480(calls):
    browser.Browser("about:blank")
    assert params(calls, "Target.createTarget") == {"url": "about:blank", "background": True}
    metrics = params(calls, "Emulation.setDeviceMetricsOverride")
    assert (metrics["width"], metrics["height"]) == (1480, 780)
    assert params(calls, "Emulation.setFocusEmulationEnabled")["enabled"] is True


def test_own_window_opens_visible_target_in_new_window(calls, monkeypatch):
    monkeypatch.setenv("JEV_OWN_WINDOW", "1")
    browser.Browser("about:blank")
    assert params(calls, "Target.createTarget") == {"url": "about:blank", "newWindow": True, "background": False}
    assert params(calls, "Emulation.setFocusEmulationEnabled")["enabled"] is True


def test_viewport_width_override(calls, monkeypatch):
    monkeypatch.setenv("JEV_VIEWPORT_WIDTH", "1120")
    assert browser.Browser("about:blank").width == 1120
    assert params(calls, "Emulation.setDeviceMetricsOverride")["width"] == 1120


@pytest.mark.parametrize("value", ["wide", "0", "-5", "1.5", ""])
def test_invalid_viewport_width_is_rejected_before_connecting(calls, monkeypatch, value):
    monkeypatch.setenv("JEV_VIEWPORT_WIDTH", value)
    with pytest.raises(ValueError, match="JEV_VIEWPORT_WIDTH"):
        browser.Browser("about:blank")
    assert calls == []


def click_expression(monkeypatch):
    calls = []
    monkeypatch.setattr(browser, "cdp", lambda method, **params: calls.append((method, params)) or {"result": {}})
    action = {"id": "e1", "kind": "click", "node": 7}
    with pytest.raises(browser.TargetRefused, match="covered"):
        browser.browser_operation({"operation": "act", "session": "s", "action": action})
    assert [method for method, _ in calls] == ["Runtime.evaluate"]
    return calls[0][1]["expression"]


def test_refused_click_is_stale_and_sends_no_input(monkeypatch):
    click_expression(monkeypatch)
    assert issubclass(browser.TargetRefused, browser.StalePage)


def test_target_is_scrolled_into_its_scroller_after_checks_and_before_the_hit_test(monkeypatch):
    js = click_expression(monkeypatch)
    order = ["nodes.get(action.node)", "isConnected", "readOnly", "scrollIntoView(", "getBoundingClientRect",
             "elementFromPoint", "action.kind==='select'"]
    positions = [js.index(part) for part in order]
    assert positions == sorted(positions)
    assert js.count("scrollIntoView(") == 1
    assert "e.scrollIntoView({block:'nearest',inline:'nearest',behavior:'instant'})" in js


NODE_DOM = """
globalThis.window = globalThis;
let scrolled = false, options = null;
const e = {
  isConnected: true, matches: () => false, closest: () => null, checkVisibility: () => true,
  scrollIntoView(o) { scrolled = true; options = o; },
  getBoundingClientRect: () => ({x: scrolled ? 1153 : 1271, y: 300, width: 40, height: 40}),
  contains: n => n === e,
};
window.__jevFast = {nodes: new Map([[7, e]])};
globalThis.innerWidth = 1480; globalThis.innerHeight = 780;
globalThis.document = {elementFromPoint: x => x < CLIP_EDGE ? e : null};
console.log(JSON.stringify({result: EXPRESSION, options}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(("clip_edge", "expected"), [(1221, {"x": 1173, "y": 320}), (1100, None)])
def test_inner_scroller_clipped_target_is_clickable_only_once_revealed(monkeypatch, tmp_path, clip_edge, expected):
    js = click_expression(monkeypatch)
    script = tmp_path / "dom.js"
    script.write_text(NODE_DOM.replace("CLIP_EDGE", str(clip_edge)).replace("EXPRESSION", js))
    output = json.loads(subprocess.run(["node", str(script)], check=True, capture_output=True, text=True).stdout)
    assert output["options"] == {"block": "nearest", "inline": "nearest", "behavior": "instant"}
    assert output["result"] == expected
