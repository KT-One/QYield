"""tui_smoke_test.py — headless end-to-end smoke test for the QYield TUI, using
Textual's Pilot test harness (no real terminal needed).

Covers:
  * Demo flow:  MainMenu -> Demo -> run -> ResultScreen.
  * Label flow: MainMenu -> Label (Phase 1: label 2 classes) -> Build ->
                ClassifyScreen (Phase 2: classify a query) -> ResultScreen
                showing Predicted + Actual.

Run: uv run python tests/tui_smoke_test.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qyield.tui import (
    ClassifyScreen, DemoScreen, LabelScreen, MainMenu, QYieldApp, ResultScreen,
)
from textual.widgets import Button, Input, Static


async def _wait_for(pilot, app, screen_cls, tries: int = 600) -> bool:
    for _ in range(tries):
        await pilot.pause(0.05)
        if isinstance(app.screen, screen_cls):
            return True
    return False


async def test_demo_flow() -> bool:
    app = QYieldApp()
    async with app.run_test(size=(120, 50)) as pilot:
        assert isinstance(app.screen, MainMenu), f"expected MainMenu, got {app.screen}"
        await pilot.click("#demo")
        await pilot.pause()
        assert isinstance(app.screen, DemoScreen), f"expected DemoScreen, got {app.screen}"

        await pilot.click("#run-demo")
        for _ in range(600):
            await pilot.pause(0.1)
            if isinstance(app.screen, ResultScreen):
                break
        assert isinstance(app.screen, ResultScreen), f"expected ResultScreen, got {app.screen}"
        print(f"[demo flow] true={app.screen.true_class} predicted={app.screen.result['predicted_class']}")
        assert app.screen.result["predicted_class"] in app.screen.result["episode_classes"]
    return True


async def test_label_flow() -> bool:
    """Phase 1: label the labelling pool across ≥2 classes (using each wafer's
    true-label hint). Build -> Phase 2: classify a query, check Predicted+Actual."""
    app = QYieldApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.click("#label")
        await pilot.pause()
        assert isinstance(app.screen, LabelScreen), f"expected LabelScreen, got {app.screen}"
        scr = app.screen

        # Label every wafer in the pool with its TRUE label (click-to-label:
        # selecting a class in the list labels the current wafer and advances).
        n = len(scr._label_pool)
        for _ in range(n):
            true_label = scr._label_pool[scr._cursor][1]
            scr._assign_current(true_label)
            await pilot.pause(0.05)
        assert len(scr._assigned_labels) == n, f"expected {n} labelled, got {len(scr._assigned_labels)}"
        assert len(set(scr._assigned_labels)) >= 2, "need >=2 classes labelled"

        # 'Continue' should now be enabled (was disabled until >=2 classes)
        assert not scr.query_one("#build", Button).disabled, "Continue should enable after >=2 classes"

        # add a custom class (verifies the free-labelling affordance)
        scr.query_one("#new-class", Input).value = "MyDefect"
        scr.query_one("#add-class", Button).press()
        await pilot.pause(0.1)
        assert "MyDefect" in scr._choices, "custom class not registered"

        # Continue -> Phase 2
        scr.query_one("#build", Button).press()
        assert await _wait_for(pilot, app, ClassifyScreen), f"expected ClassifyScreen, got {app.screen}"
        await pilot.pause(0.2)
        csr = app.screen
        print(f"[label flow] built support set; {len(csr._queries)} queries queued")

        # classify the current query -> inline verdict (Predicted + Actual), Next enabled
        csr.query_one("#classify", Button).press()
        for _ in range(600):
            await pilot.pause(0.1)
            if not csr.query_one("#next", Button).disabled:
                break
        assert csr._classified, "wafer should be classified"
        verdict = str(csr.query_one("#verdict", Static).renderable)
        assert "Predicted:" in verdict and "Actual:" in verdict, f"verdict missing fields: {verdict!r}"
        assert not csr.query_one("#next", Button).disabled, "Next should be enabled after Classify"
        print("[label flow] inline verdict:\n    " + verdict.replace("\n", "\n    "))

        # Next advances to the following query and resets the verdict
        prev_cursor = csr._cursor
        csr.query_one("#next", Button).press()
        await pilot.pause(0.2)
        assert csr._cursor == prev_cursor + 1, "Next should advance the query cursor"
        assert csr.query_one("#next", Button).disabled, "Next should reset (disabled) after advancing"
        print(f"[label flow] advanced to query {csr._cursor + 1}/{len(csr._queries)}")
    return True


async def main() -> int:
    print("QYield TUI smoke test\n" + "=" * 40)
    ok1 = await test_demo_flow()
    ok2 = await test_label_flow()
    ok = ok1 and ok2
    print("\nAll TUI flows completed without error." if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
