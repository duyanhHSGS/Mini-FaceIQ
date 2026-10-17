"""Regression tests for no-rebuild navigation and bounded image previews."""

from __future__ import annotations

import inspect
import io
from types import SimpleNamespace
from unittest.mock import patch

import flet as ft
from PIL import Image
from test_android_manual_ui import FakePage, app_main, walk


def _jpeg(color: tuple[int, int, int]) -> bytes:
    with io.BytesIO() as output:
        Image.new("RGB", (2400, 1600), color).save(output, "JPEG", quality=92)
        return output.getvalue()


def _button(page: FakePage, label: str) -> ft.Button:
    matches = [
        item
        for item in walk(page.controls[0])
        if isinstance(item, ft.Button) and item.content == label and item.visible
    ]
    # One cached button can exist on Align and another on Results.
    # The callback is identical for either and updates the active view.
    assert matches, f"Missing button: {label}"
    return matches[0]


def _tap(page: FakePage, name: str) -> None:
    callback = _button(page, name).on_click
    assert callback is not None
    callback(None)  # pyright: ignore[reportArgumentType, reportCallIssue]


def _state(page: FakePage):
    callback = _button(page, "Front view").on_click
    assert callback is not None
    handler = inspect.getclosurevars(callback).nonlocals["switch_profile"]
    render = inspect.getclosurevars(handler).nonlocals["render"]
    namespace = inspect.getclosurevars(render).nonlocals
    return render, namespace


def _add_report(page: FakePage) -> None:
    render, _ = _state(page)
    workspace = inspect.getclosurevars(render).nonlocals["workspace"]
    responses = inspect.getclosurevars(workspace).nonlocals["responses"]
    for profile, number in (("front", 48), ("side", 31)):
        metrics = [
            {
                "name": f"Metric {i}",
                "value": float(i),
                "unit": "mm",
                "score": 7.0,
                "category": "geometry",
            }
            for i in range(number)
        ]
        responses[profile] = {
            "landmark_count": number,
            "data": {
                "overallScore": 7.2,
                "categoryScores": {"symmetry": 6.9},
                "frontMeasurements": metrics,
                "sideMeasurements": metrics,
            },
        }
    render()


def _setup() -> FakePage:
    page = FakePage()
    app_main(page)
    for index, (profile, color) in enumerate(
        (("front", (10, 40, 90)), ("side", (20, 70, 100)))
    ):
        page.services[index].on_result(
            SimpleNamespace(
                files=[SimpleNamespace(name=f"{profile}.jpg", bytes=_jpeg(color))]
            )
        )
    assert page.navigation_bar is not None
    assert page.navigation_bar.on_change is not None
    page.navigation_bar.on_change(
        SimpleNamespace(control=SimpleNamespace(selected_index=2))
    )
    _add_report(page)
    return page


def test_repeated_front_side_switch_reuses_widgets_without_rebuilding() -> None:
    page = _setup()
    _tap(page, "Side view")
    _tap(page, "Front view")
    _render, state = _state(page)
    sidebar_views = state["sidebar_views"]
    workspace_views = state["workspace_views"]
    front_sidebar = sidebar_views[("results", "front")].content
    side_sidebar = sidebar_views[("results", "side")].content
    front_workspace = workspace_views[("results", "front")].content
    side_workspace = workspace_views[("results", "side")].content
    ids = tuple(map(id, (front_sidebar, side_sidebar, front_workspace, side_workspace)))
    before_all = {id(control) for control in walk(page.controls[0])}

    # The expensive constructors must not run when switching cached views.
    import main as app_module
    _photo_stage = app_module.__dict__["_photo_stage"]
    _report_body = app_module.__dict__["_report_body"]

    with (
        patch("main._photo_stage", wraps=_photo_stage) as photos,
        patch("main._report_body", wraps=_report_body) as reports,
    ):
        for _ in range(60):
            _tap(page, "Side view")
            _tap(page, "Front view")
        assert photos.call_count == 0
        assert reports.call_count == 0

    after_all = {id(control) for control in walk(page.controls[0])}
    assert before_all == after_all
    assert ids == tuple(
        map(
            id,
            (
                sidebar_views[("results", "front")].content,
                sidebar_views[("results", "side")].content,
                workspace_views[("results", "front")].content,
                workspace_views[("results", "side")].content,
            ),
        )
    )


def test_preview_is_smaller_but_analysis_still_has_original_photo() -> None:
    page = _setup()
    _, state = _state(page)
    photos = inspect.getclosurevars(state["workspace"]).nonlocals["photos"]
    previews = inspect.getclosurevars(state["workspace"]).nonlocals["previews"]
    for profile in ("front", "side"):
        assert photos[profile][1] != previews[profile]
        with Image.open(io.BytesIO(previews[profile])) as image:
            assert image.width <= 960
            assert image.height <= 960
    visible_images = [
        item for item in walk(page.controls[0]) if isinstance(item, ft.Image)
    ]
    assert visible_images
    assert all(item.cache_width is not None for item in visible_images)


def test_replacing_photo_invalidates_only_affected_view() -> None:
    page = _setup()
    _tap(page, "Side view")
    _tap(page, "Front view")
    _render, state = _state(page)
    views = state["workspace_views"]
    old_front = views[("results", "front")].content.controls
    old_side = views[("results", "side")].content.controls
    page.services[0].on_result(
        SimpleNamespace(
            files=[SimpleNamespace(name="new_front.jpg", bytes=_jpeg((200, 25, 30)))]
        )
    )
    assert views[("results", "front")].content.controls is not old_front
    assert views[("results", "side")].content.controls is old_side
    _tap(page, "Side view")
    assert views[("results", "side")].content.controls is old_side


def test_reselecting_active_view_is_zero_updates() -> None:
    page = _setup()
    _tap(page, "Front view")
    current = page.update_count
    _tap(page, "Front view")
    assert page.update_count == current

def _go_tab(page: FakePage, index: int) -> None:
    navigation = page.navigation_bar
    assert isinstance(navigation, ft.NavigationBar)
    callback = navigation.on_change
    assert callback is not None
    callback(SimpleNamespace(control=SimpleNamespace(selected_index=index)))


def test_navigation_across_all_tabs_reuses_mounted_views() -> None:
    page = _setup()
    # Warm each tab/profile once; subsequent navigation only toggles visibility.
    _go_tab(page, 0)
    _go_tab(page, 1)
    _tap(page, "Front view")
    _tap(page, "Side view")
    _go_tab(page, 2)
    _tap(page, "Front view")
    _tap(page, "Side view")
    _go_tab(page, 3)

    import main as app_module

    with (
        patch("main._photo_stage", wraps=app_module.__dict__["_photo_stage"]) as photos,
        patch("main._report_body", wraps=app_module.__dict__["_report_body"]) as reports,
    ):
        for _ in range(25):
            _go_tab(page, 0)
            _go_tab(page, 1)
            _tap(page, "Front view")
            _tap(page, "Side view")
            _go_tab(page, 2)
            _tap(page, "Front view")
            _tap(page, "Side view")
            _go_tab(page, 3)
        assert photos.call_count == 0
        assert reports.call_count == 0


def test_sort_change_invalidates_active_report_only() -> None:
    page = _setup()
    _tap(page, "Front view")
    _tap(page, "Side view")
    _, state = _state(page)
    workspaces = state["workspace_views"]
    old_front = workspaces[("results", "front")].content.controls
    old_side = workspaces[("results", "side")].content.controls
    sorted_drop = next(
        (item for item in walk(page.controls[0])
         if isinstance(item, ft.Dropdown) and item.label == "Sort measurements"),
        None,
    )
    assert sorted_drop is not None
    callback = sorted_drop.on_select
    assert callback is not None
    callback(SimpleNamespace(control=SimpleNamespace(value="high")))  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert workspaces[("results", "side")].content.controls is not old_side
    assert workspaces[("results", "front")].content.controls is old_front


def test_preview_preserves_portrait_aspect_ratio() -> None:
    from main import _image_preview  # pyright: ignore[reportAttributeAccessIssue]

    portrait = Image.new("RGB", (3600, 2400), (20, 30, 70))
    with io.BytesIO() as output:
        portrait.save(output, format="JPEG", quality=95)
        original = output.getvalue()
    resized = _image_preview(original)
    with Image.open(io.BytesIO(resized)) as picture:
        assert picture.size == (960, 640)
    assert len(resized) < len(original)


def test_normal_labels_use_lightweight_flutter_text() -> None:
    import main as app_module

    make_text = app_module.__dict__["_text"]
    assert make_text("Plain label").selectable is False
    assert make_text("Copyable summary", selectable=True).selectable is True
