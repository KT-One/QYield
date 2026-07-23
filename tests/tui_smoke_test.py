"""tui_smoke_test.py — headless end-to-end smoke test for the QYield TUI, using
Textual's Pilot test harness (no real terminal needed). Drives: MainMenu -> Demo
-> pick a class -> run prediction -> ResultScreen; then MainMenu -> Upload ->
enter a path -> load preview -> run prediction -> ResultScreen.

Run: uv run python tests/tui_smoke_test.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qyield.constants import DEFAULT_KSET_PATH
from qyield.model import REPO_ROOT, load_kset
from qyield.tui import DemoScreen, MainMenu, QYieldApp, ResultScreen, UploadScreen


async def test_demo_flow() -> bool:
    app = QYieldApp()
    async with app.run_test(size=(120, 50)) as pilot:
        assert isinstance(app.screen, MainMenu), f"expected MainMenu, got {app.screen}"
        await pilot.click("#demo")
        await pilot.pause()
        assert isinstance(app.screen, DemoScreen), f"expected DemoScreen, got {app.screen}"

        run_btn = app.screen.query_one("#run-demo")
        await pilot.click("#run-demo")
        # wait for the background worker (model load + inference) to finish
        for _ in range(600):   # up to ~60s
            await pilot.pause(0.1)
            if isinstance(app.screen, ResultScreen):
                break
        if not isinstance(app.screen, ResultScreen):
            status = app.screen.query_one("#demo-status")
            print(f"[demo flow] status widget content: {status.renderable!r}")
        assert isinstance(app.screen, ResultScreen), f"expected ResultScreen, got {app.screen}"
        print(f"[demo flow] true={app.screen.true_class} predicted={app.screen.result['predicted_class']}")
        assert app.screen.result["predicted_class"] in app.screen.result["episode_classes"]
    return True


async def test_upload_flow() -> bool:
    imgs, labels, classes = load_kset(REPO_ROOT / DEFAULT_KSET_PATH)
    tmp_path = Path("/tmp/qyield_tui_smoke_query.npy")
    np.save(tmp_path, imgs[10])   # continuous [0,1] float, as predict_array would accept
    true_label = str(labels[10])

    app = QYieldApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.click("#upload")
        await pilot.pause()
        assert isinstance(app.screen, UploadScreen), f"expected UploadScreen, got {app.screen}"

        path_input = app.screen.query_one("#path-input")
        path_input.value = str(tmp_path)
        await pilot.click("#load-preview")
        await pilot.pause(0.2)

        run_btn = app.screen.query_one("#run-upload")
        assert not run_btn.disabled, "run-upload should be enabled after a successful preview load"

        await pilot.click("#run-upload")
        for _ in range(600):
            await pilot.pause(0.1)
            if isinstance(app.screen, ResultScreen):
                break
        assert isinstance(app.screen, ResultScreen), f"expected ResultScreen, got {app.screen}"
        print(f"[upload flow] true={true_label} predicted={app.screen.result['predicted_class']}")
        assert app.screen.result["predicted_class"] in app.screen.result["episode_classes"]

    tmp_path.unlink(missing_ok=True)
    return True


async def main() -> int:
    print("QYield TUI smoke test\n" + "=" * 40)
    ok1 = await test_demo_flow()
    ok2 = await test_upload_flow()
    print("\nAll TUI flows completed without error." if (ok1 and ok2) else "\nFAILED")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
