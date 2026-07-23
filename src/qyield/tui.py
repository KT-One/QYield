"""tui.py — Textual-based interactive interface for QYield.

Run: `qyield tui`

Screens:
  MainMenu    — choose Demo (bundled K-set) or Upload (your own wafer map).
  DemoScreen  — pick a class from the bundled K-set, preview it, run inference.
  UploadScreen — enter a path (or launch a native OS file picker if available),
                 preview it, run inference.
  ResultScreen — predicted class + full ranking table, shared by both flows.

The model loads lazily (on first prediction) and is cached for the app's lifetime
— so switching between screens/predictions doesn't reload the checkpoint each time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, Static

from .constants import ALL_DEFECT_CLASSES, DEFAULT_KSET_PATH
from .model import REPO_ROOT, QYieldModel, load_kset, load_query_image
from .native_picker import native_picker_available, pick_file_native
from .wafer_render import render_wafer_ansi


class ModelHolder:
    """Lazily-constructed, app-lifetime-cached QYieldModel + K-set. QYieldModel
    itself transparently falls back to CPU if GPU inference isn't supported, so
    no device handling is needed here."""

    def __init__(self) -> None:
        self._model: QYieldModel | None = None
        self._kset = None

    def get_model(self) -> QYieldModel:
        if self._model is None:
            self._model = QYieldModel()
        return self._model

    def get_kset(self):
        if self._kset is None:
            self._kset = load_kset(REPO_ROOT / DEFAULT_KSET_PATH)
        return self._kset


class ResultScreen(Screen):
    """Shows a wafer preview + the prediction ranking. Pushed on top of whichever
    screen triggered it; Escape/Back returns there."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, preview: str, true_class: str | None, result: dict) -> None:
        super().__init__()
        self.preview = preview
        self.true_class = true_class
        self.result = result

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with VerticalScroll(classes="panel"):
                yield Label("Wafer preview", classes="title")
                yield Static(self.preview, classes="wafer-preview")
                if self.true_class:
                    yield Label(f"True class: {self.true_class}", classes="subtitle")
            with VerticalScroll(classes="panel"):
                yield Label(f"Predicted: {self.result['predicted_class']}", classes="title predicted")
                yield Label(f"Episode classes: {', '.join(self.result['episode_classes'])}",
                           classes="subtitle")
                table = DataTable(id="ranking")
                table.add_columns("Class", "Distance")
                for cls, dist in self.result["ranking"]:
                    marker = " <-- predicted" if cls == self.result["predicted_class"] else ""
                    table.add_row(cls, f"{dist:.3f}{marker}")
                yield table
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ranking", DataTable).focus()


class DemoScreen(Screen):
    """Pick a class from the bundled K-set, preview a sample image, run inference."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Demo — classify a sample from our bundled K-set", classes="title")
        yield Label("Pick a class (or Any), then Run:", classes="subtitle")
        with Horizontal():
            yield ListView(
                ListItem(Label("Any (random)"), id="any"),
                *[ListItem(Label(c), id=f"cls-{c}") for c in ALL_DEFECT_CLASSES],
                id="class-list",
            )
            with VerticalScroll(classes="panel"):
                yield Button("Run prediction", id="run-demo", variant="primary")
                yield Static("", id="demo-status")
                yield Static("", id="demo-preview", classes="wafer-preview")
        yield Footer()

    def on_mount(self) -> None:
        self._selected_class: str | None = None
        self._refresh_preview()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        self._selected_class = None if item_id == "any" else item_id.removeprefix("cls-")
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        holder: ModelHolder = self.app.holder  # type: ignore[attr-defined]
        imgs, labels, classes = holder.get_kset()
        labels = np.asarray(labels)
        rng = np.random.default_rng()
        if self._selected_class:
            idx = np.where(labels == self._selected_class)[0]
        else:
            idx = np.arange(len(imgs))
        self._preview_idx = int(rng.choice(idx))
        self._preview_label = str(labels[self._preview_idx])
        self.query_one("#demo-preview", Static).update(render_wafer_ansi(imgs[self._preview_idx]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-demo":
            self._run()

    @work(exclusive=True, thread=True)
    def _run(self) -> None:
        status = self.query_one("#demo-status", Static)
        self.app.call_from_thread(status.update, "Running inference...")
        holder: ModelHolder = self.app.holder  # type: ignore[attr-defined]
        try:
            model = holder.get_model()
            imgs, labels, classes = holder.get_kset()
            img = imgs[self._preview_idx]
            result = model.predict_array(img)
        except Exception as exc:
            self.app.call_from_thread(status.update, f"Error: {exc}")
            return
        preview = render_wafer_ansi(img)
        self.app.call_from_thread(status.update, "")
        self.app.call_from_thread(
            self.app.push_screen, ResultScreen(preview, self._preview_label, result)
        )


class UploadScreen(Screen):
    """Enter a path (or launch a native file picker), preview it, run inference."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Upload — classify your own wafer map", classes="title")
        yield Label(".npy (raw {0,1,2}, recommended) or .png/.jpg", classes="subtitle")
        with Horizontal():
            yield Input(placeholder="/path/to/your_wafer.npy", id="path-input")
            picker_label = "Browse..." if native_picker_available() else "Browse (unavailable)"
            yield Button(picker_label, id="browse", disabled=not native_picker_available())
        with Horizontal(classes="actions"):
            yield Button("Load + preview", id="load-preview")
            yield Button("Run prediction", id="run-upload", variant="primary", disabled=True)
        yield Static("", id="upload-status")
        with VerticalScroll(classes="panel"):
            yield Static("", id="upload-preview", classes="wafer-preview")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse":
            self._browse()
        elif event.button.id == "load-preview":
            self._load_preview()
        elif event.button.id == "run-upload":
            self._run()

    @work(exclusive=True, thread=True)
    def _browse(self) -> None:
        path = pick_file_native("Select a wafer map (.npy/.png/.jpg)")
        if path:
            self.app.call_from_thread(self.query_one("#path-input", Input).__setattr__, "value", path)
            self.app.call_from_thread(self._load_preview)

    def _load_preview(self) -> None:
        path_str = self.query_one("#path-input", Input).value.strip()
        status = self.query_one("#upload-status", Static)
        run_btn = self.query_one("#run-upload", Button)
        if not path_str:
            status.update("Enter a path first.")
            return
        p = Path(path_str).expanduser()
        if not p.exists():
            status.update(f"File not found: {p}")
            run_btn.disabled = True
            return
        try:
            img = load_query_image(p, img_size=224)
        except Exception as exc:
            status.update(f"Error loading file: {exc}")
            run_btn.disabled = True
            return
        self._loaded_path = p
        self._loaded_img = img
        self.query_one("#upload-preview", Static).update(render_wafer_ansi(img))
        status.update(f"Loaded {p.name} — ready to run.")
        run_btn.disabled = False

    @work(exclusive=True, thread=True)
    def _run(self) -> None:
        status = self.query_one("#upload-status", Static)
        self.app.call_from_thread(status.update, "Running inference...")
        holder: ModelHolder = self.app.holder  # type: ignore[attr-defined]
        try:
            model = holder.get_model()
            result = model.predict_array(self._loaded_img)
        except Exception as exc:
            self.app.call_from_thread(status.update, f"Error: {exc}")
            return
        preview = render_wafer_ansi(self._loaded_img)
        self.app.call_from_thread(status.update, "")
        self.app.call_from_thread(
            self.app.push_screen, ResultScreen(preview, None, result)
        )


class MainMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="panel centered"):
            yield Label("QYield", classes="app-title")
            yield Label("Quantum wafer-defect classifier", classes="subtitle")
            yield Button("Demo — try a sample from our K-set", id="demo", variant="primary")
            yield Button("Upload — classify your own wafer map", id="upload", variant="primary")
            yield Button("Quit", id="quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "demo":
            self.app.push_screen(DemoScreen())
        elif event.button.id == "upload":
            self.app.push_screen(UploadScreen())
        elif event.button.id == "quit":
            self.app.exit()


class QYieldApp(App):
    CSS = """
    .title { text-style: bold; padding: 1 0 0 1; }
    .app-title { text-style: bold; padding: 1 0 0 1; content-align: center middle; width: 100%; }
    .subtitle { color: $text-muted; padding: 0 0 1 1; }
    .predicted { color: $success; }
    .panel { border: round $primary; padding: 1; margin: 1; }
    .centered { align: center middle; }
    .wafer-preview { border: round $secondary; padding: 1; min-height: 10; }
    Button { margin: 1 0; width: 100%; }
    #class-list { width: 30%; }
    .actions { height: auto; }
    .actions Button { width: 1fr; margin: 1 1; }
    """
    TITLE = "QYield"

    def on_mount(self) -> None:
        self.holder = ModelHolder()
        self.push_screen(MainMenu())


def run() -> None:
    QYieldApp().run()


if __name__ == "__main__":
    run()
