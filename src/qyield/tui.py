"""tui.py — Textual-based interactive interface for QYield.

Run: `qyield tui`

Screens:
  MainMenu          — Demo, or Label-your-own (2-phase).
  DemoScreen        — pick a class from the bundled K-set, preview it, classify.
  LabelScreen       — PHASE 1: step through a few wafers, each showing its TRUE
                      label as a hint; assign whatever class you like to build a
                      custom support set.
  ClassifyScreen    — PHASE 2: step through held-out query wafers, classify each
                      against the support set you just built; ResultScreen shows
                      Predicted AND Actual.
  ResultScreen      — wafer preview + predicted class + ranking (+ true class).

Previews render as green/red ANSI half-blocks (green = good die, red = defect).
Class names are shown with friendly DISPLAY_NAMES; internal keys stay the source
of truth.
"""
from __future__ import annotations

import numpy as np
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, Static

from .constants import ALL_DEFECT_CLASSES, DEFAULT_KSET_PATH, display_name
from .model import REPO_ROOT, load_kset
from .model_l4 import QYieldL4Model
from .preview import make_preview_widget, update_preview_widget


class ModelHolder:
    """Lazily-constructed, app-lifetime-cached QYieldL4Model + K-set. The model
    itself transparently falls back to CPU if GPU inference isn't supported."""

    def __init__(self) -> None:
        self._model: QYieldL4Model | None = None
        self._kset = None

    def get_model(self) -> QYieldL4Model:
        if self._model is None:
            self._model = QYieldL4Model()
        return self._model

    def get_kset(self):
        if self._kset is None:
            self._kset = load_kset(REPO_ROOT / DEFAULT_KSET_PATH)
        return self._kset


class ResultScreen(Screen):
    """Wafer preview + prediction ranking (+ true class). Escape returns back."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, wafer: np.ndarray, true_class: str | None, result: dict) -> None:
        super().__init__()
        self.wafer = wafer
        self.true_class = true_class
        self.result = result

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with VerticalScroll(classes="panel"):
                yield Label("Wafer preview", classes="title")
                yield make_preview_widget(self.wafer)
                if self.true_class is not None:
                    correct = self.true_class == self.result["predicted_class"]
                    mark = "✓" if correct else "✗"
                    yield Label(f"Actual: {display_name(self.true_class)}  {mark}",
                               classes="subtitle " + ("predicted" if correct else "wrong"))
            with VerticalScroll(classes="panel"):
                yield Label(f"Predicted: {display_name(self.result['predicted_class'])}",
                           classes="title predicted")
                yield Label("Episode: " + ", ".join(display_name(c) for c in self.result["episode_classes"]),
                           classes="subtitle")
                table = DataTable(id="ranking")
                table.add_columns("Class", "Distance")
                for cls, dist in self.result["ranking"]:
                    mk = " ← predicted" if cls == self.result["predicted_class"] else ""
                    table.add_row(display_name(cls), f"{dist:.3f}{mk}")
                yield table
        yield Footer()

    def on_mount(self) -> None:
        # focus the ranking table once children are mounted; best-effort (cosmetic
        # keyboard-nav convenience — must never crash the result screen).
        self.call_after_refresh(self._focus_ranking)

    def _focus_ranking(self) -> None:
        tables = list(self.query(DataTable))
        if tables:
            tables[0].focus()


class DemoScreen(Screen):
    """Pick a class from the bundled K-set, preview a sample, classify it."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, holder: ModelHolder) -> None:
        super().__init__()
        self.holder = holder
        self._imgs, labels, _ = holder.get_kset()
        self._labels = np.asarray(labels)
        self._selected_class: str | None = None
        self._pick()

    def _pick(self) -> None:
        rng = np.random.default_rng()
        idx = (np.where(self._labels == self._selected_class)[0] if self._selected_class
               else np.arange(len(self._imgs)))
        self._preview_idx = int(rng.choice(idx))
        self._preview_label = str(self._labels[self._preview_idx])

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Demo — classify a sample from our bundled K-set", classes="title")
        yield Label("Pick a class (or Any), then Run:", classes="subtitle")
        with Horizontal():
            yield ListView(
                ListItem(Label("Any (random)"), id="any"),
                *[ListItem(Label(display_name(c)), id=f"cls-{c}") for c in ALL_DEFECT_CLASSES],
                id="class-list",
            )
            with VerticalScroll(classes="panel"):
                yield Button("Classify", id="run-demo", variant="primary")
                yield Static("", id="demo-status")
                prev = make_preview_widget(self._imgs[self._preview_idx])
                prev.id = "demo-preview"
                yield prev
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        self._selected_class = None if item_id == "any" else item_id.removeprefix("cls-")
        self._pick()
        update_preview_widget(self.query_one("#demo-preview"), self._imgs[self._preview_idx])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-demo":
            self._run()

    @work(exclusive=True, thread=True)
    def _run(self) -> None:
        self.app.call_from_thread(setattr, self, "loading", True)
        try:
            model = self.holder.get_model()
            model.reset_support_set()      # demo uses the bundled K-set
            img = self._imgs[self._preview_idx]
            result = model.predict_array(img)
        except Exception as exc:
            self.app.call_from_thread(setattr, self, "loading", False)
            self.app.call_from_thread(self.query_one("#demo-status", Static).update, f"Error: {exc}")
            return
        self.app.call_from_thread(setattr, self, "loading", False)
        self.app.call_from_thread(self.app.push_screen,
                                  ResultScreen(img, self._preview_label, result))


class LabelScreen(Screen):
    """PHASE 1 — label a few wafers to build your own support set.

    Draws a small labelling pool (a few classes x a couple shots) from the
    bundled K-set. Each wafer shows its TRUE label as a hint, but you may assign
    any class you like (including brand-new custom classes). When ≥2 classes have
    been labelled, move on to Phase 2 (classify)."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, holder: ModelHolder, n_classes: int = 3, shots: int = 2,
                 n_queries: int = 3) -> None:
        super().__init__()
        self.holder = holder
        imgs, labels, _ = holder.get_kset()
        labels = np.asarray(labels)
        rng = np.random.default_rng()

        present = [c for c in ALL_DEFECT_CLASSES if (labels == c).sum() >= shots + 1]
        chosen = [str(c) for c in rng.choice(present, size=min(n_classes, len(present)), replace=False)]

        label_pool, query_pool = [], []
        for c in chosen:
            idx = list(rng.permutation(np.where(labels == c)[0]))
            for i in idx[:shots]:
                label_pool.append((imgs[i], c))
            for i in idx[shots:shots + max(1, n_queries // len(chosen) + 1)]:
                query_pool.append((imgs[i], c))
        rng.shuffle(label_pool)
        rng.shuffle(query_pool)

        self._label_pool = label_pool
        self._query_pool = query_pool[:n_queries] if n_queries else query_pool
        self._cursor = 0
        self._assigned_imgs: list[np.ndarray] = []
        self._assigned_labels: list[str] = []
        self._choices: list[str] = list(ALL_DEFECT_CLASSES)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Label your own — Phase 1 of 2: label a few wafers", classes="title")
        yield Label("Click the class each wafer belongs to (the true label is shown as a hint). "
                    "Add your own class names if you like. Label ≥2 classes, then Continue.",
                    classes="subtitle")
        first_img, first_true = self._label_pool[0]
        with Horizontal():
            with VerticalScroll(classes="sidebar"):
                yield Label("Click a class to label →", classes="subtitle")
                yield ListView(*[ListItem(Label(display_name(c)), id=f"opt-{c}") for c in self._choices],
                               id="choice-list")
                yield Input(placeholder="add your own class…", id="new-class")
                yield Button("Add", id="add-class")
                yield Button("Skip", id="skip")
                yield Button("Continue", id="build", variant="success", disabled=True)
                yield Static("", id="label-status")
            with VerticalScroll(classes="panel stage"):
                yield Label(f"Wafer 1/{len(self._label_pool)}   ·   labelled 0 across 0 class(es)",
                           id="label-progress", classes="subtitle")
                yield Label(f"Hint — true label: {display_name(first_true)}",
                           id="label-hint", classes="hint")
                prev = make_preview_widget(first_img)
                prev.id = "label-preview"
                yield prev
        yield Footer()

    def _refresh(self) -> None:
        n_classes = len(set(self._assigned_labels))
        prog = self.query_one("#label-progress", Static)
        hint = self.query_one("#label-hint", Static)
        if self._cursor < len(self._label_pool):
            img, true = self._label_pool[self._cursor]
            prog.update(f"Wafer {self._cursor + 1}/{len(self._label_pool)}   ·   "
                        f"labelled {len(self._assigned_labels)} across {n_classes} class(es)")
            hint.update(f"Hint — true label: {display_name(true)}")
            update_preview_widget(self.query_one("#label-preview"), img)
        else:
            prog.update(f"All {len(self._label_pool)} wafers seen · "
                        f"labelled {len(self._assigned_labels)} across {n_classes} class(es)")
            hint.update("Done — click Continue to classify." if n_classes >= 2
                        else "Label at least 2 classes to continue.")

    def _update_continue(self) -> None:
        self.query_one("#build", Button).disabled = len(set(self._assigned_labels)) < 2

    def _assign_current(self, cls: str) -> None:
        """Label the current wafer with `cls` and advance."""
        if self._cursor >= len(self._label_pool):
            self.query_one("#label-status", Static).update("All wafers labelled — click Continue.")
            return
        self._assigned_imgs.append(self._label_pool[self._cursor][0])
        self._assigned_labels.append(cls)
        self.query_one("#label-status", Static).update(f"Labelled as '{display_name(cls)}'.")
        self._cursor += 1
        self._refresh()
        self._update_continue()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        cls = (event.item.id or "").removeprefix("opt-")
        if cls:
            self._assign_current(cls)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        status = self.query_one("#label-status", Static)
        if bid == "add-class":
            name = self.query_one("#new-class", Input).value.strip()
            if not name:
                status.update("Type a class name first.")
                return
            if name not in self._choices:
                self._choices.append(name)
                self.query_one("#choice-list", ListView).append(
                    ListItem(Label(display_name(name)), id=f"opt-{name}"))
            self.query_one("#new-class", Input).value = ""
            status.update(f"Added '{name}'. Click it in the list to label this wafer.")
        elif bid == "skip":
            if self._cursor < len(self._label_pool):
                self._cursor += 1
                self._refresh()
        elif bid == "build":
            if len(set(self._assigned_labels)) < 2:
                status.update("Label at least 2 different classes first.")
                return
            self._build()

    @work(exclusive=True, thread=True)
    def _build(self) -> None:
        self.app.call_from_thread(setattr, self, "loading", True)
        try:
            model = self.holder.get_model()
            model.build_support_set(np.stack(self._assigned_imgs), list(self._assigned_labels))
        except Exception as exc:
            self.app.call_from_thread(setattr, self, "loading", False)
            self.app.call_from_thread(self.query_one("#label-status", Static).update, f"Error: {exc}")
            return
        self.app.call_from_thread(setattr, self, "loading", False)
        self.app.call_from_thread(self.app.push_screen, ClassifyScreen(self.holder, self._query_pool))


class ClassifyScreen(Screen):
    """PHASE 2 — classify held-out query wafers against the support set built in
    Phase 1. The verdict (Predicted vs Actual) is shown inline; a Next button
    appears once the current wafer has been classified."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, holder: ModelHolder, queries: list) -> None:
        super().__init__()
        self.holder = holder
        self._queries = [(img, str(true)) for img, true in queries]   # list of (img, true_label)
        self._cursor = 0
        self._classified = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Label your own — Phase 2 of 2: classify", classes="title")
        yield Label("Held-out wafers classified using ONLY the labels you assigned.",
                    classes="subtitle")
        with Horizontal():
            with VerticalScroll(classes="sidebar"):
                yield Button("Classify", id="classify", variant="primary")
                yield Button("Next", id="next", variant="success", disabled=True)
                yield Static("", id="verdict")
            with VerticalScroll(classes="panel stage"):
                yield Label(f"Query 1/{len(self._queries)}" if self._queries else "(no queries)",
                           id="classify-progress", classes="subtitle")
                prev = make_preview_widget(self._queries[0][0]) if self._queries else Static("(no queries)")
                prev.id = "classify-preview"
                yield prev
        yield Footer()

    def _show_query(self) -> None:
        prog = self.query_one("#classify-progress", Static)
        if self._cursor < len(self._queries):
            prog.update(f"Query {self._cursor + 1}/{len(self._queries)}")
            update_preview_widget(self.query_one("#classify-preview"), self._queries[self._cursor][0])
        else:
            prog.update("All queries classified — Esc to go back.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "classify":
            if not self._classified and self._cursor < len(self._queries):
                self._run()
        elif event.button.id == "next":
            self._advance()

    def _advance(self) -> None:
        if self._cursor >= len(self._queries) - 1:
            self.query_one("#verdict", Static).update("[dim]No more queries — Esc to go back.[/dim]")
            self.query_one("#next", Button).disabled = True
            return
        self._cursor += 1
        self._classified = False
        self.query_one("#verdict", Static).update("")
        self.query_one("#next", Button).disabled = True
        self.query_one("#classify", Button).disabled = False
        self._show_query()

    @work(exclusive=True, thread=True)
    def _run(self) -> None:
        self.app.call_from_thread(setattr, self, "loading", True)
        img, true = self._queries[self._cursor]
        try:
            model = self.holder.get_model()   # active support set = the user's (Phase 1)
            result = model.predict_array(img)
        except Exception as exc:
            self.app.call_from_thread(setattr, self, "loading", False)
            self.app.call_from_thread(self.query_one("#verdict", Static).update, f"Error: {exc}")
            return
        pred = result["predicted_class"]
        correct = pred == true
        lines = [
            f"[b]Predicted:[/b] {display_name(pred)}",
            f"[b]Actual:[/b] {display_name(true)}  " + ("[green]✓[/green]" if correct else "[red]✗[/red]"),
            "",
            "[dim]nearest prototypes[/dim]",
        ]
        for c, d in result["ranking"][:5]:
            mk = " [green]←[/green]" if c == pred else ""
            lines.append(f"  {display_name(c)}: {d:.2f}{mk}")
        verdict = "\n".join(lines)

        def _apply() -> None:
            self.loading = False
            self._classified = True
            self.query_one("#verdict", Static).update(verdict)
            self.query_one("#classify", Button).disabled = True
            self.query_one("#next", Button).disabled = False
        self.app.call_from_thread(_apply)


class MainMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="panel centered"):
            yield Label("QYield", classes="app-title")
            yield Button("Run Demo", id="demo", variant="primary")
            yield Button("Try yourself", id="label", variant="primary")
            yield Button("Quit", id="quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        holder = self.app.holder  # type: ignore[attr-defined]
        if event.button.id == "demo":
            self.app.push_screen(DemoScreen(holder))
        elif event.button.id == "label":
            self.app.push_screen(LabelScreen(holder))
        elif event.button.id == "quit":
            self.app.exit()


class QYieldApp(App):
    CSS = """
    .title { text-style: bold; padding: 1 0 0 1; }
    .app-title { text-style: bold; padding: 1 0 0 1; content-align: center middle; width: 100%; }
    .subtitle { color: $text-muted; padding: 0 0 1 1; }
    .menu-desc { color: $text-muted; padding: 0 0 1 3; }
    .hint { color: $warning; padding: 0 0 1 1; }
    .predicted { color: $success; }
    .wrong { color: $error; }
    .panel { border: round $primary; padding: 1; margin: 1; }
    .stage { width: 2fr; }
    .sidebar { width: 1fr; padding: 0 1; }
    .centered { align: center middle; }
    .wafer-preview { border: round $secondary; padding: 1; min-height: 10; }
    Button { margin: 1 0; width: 100%; }
    #class-list { width: 30%; }
    #choice-list { height: 10; }
    """
    TITLE = "QYield"

    def on_mount(self) -> None:
        self.holder = ModelHolder()
        self.push_screen(MainMenu())


def run() -> None:
    QYieldApp().run()


if __name__ == "__main__":
    run()
