from __future__ import annotations

import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "script/run_alpha360_e0_e6_finalizer_launchd.sh"
PLIST = ROOT / "script/com.hmax.qlib.alpha360-e0-e6-finalize.plist"


def test_launchd_finalizer_is_short_lived_and_self_unloads() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "sleep" not in source
    assert "while" not in source
    assert "nohup" not in source
    assert "launchctl bootout" in source
    assert "status -eq 3" in source


def test_launchd_interval_is_thirty_minutes_and_low_priority() -> None:
    with PLIST.open("rb") as stream:
        value = plistlib.load(stream)
    assert value["Label"] == "com.hmax.qlib.alpha360-e0-e6-finalize"
    assert value["StartInterval"] == 1800
    assert value["RunAtLoad"] is True
    assert value["LowPriorityIO"] is True
    assert value["Nice"] == 10
