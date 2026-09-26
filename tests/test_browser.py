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
