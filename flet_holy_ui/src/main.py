"""MINIFACEIQ_MATERIAL3_ADAPTIVE: Android-style responsive Flet workspace.

The Flet application calls the existing Python analysis engine in-process.
It does not start Flask, upload photos, or require a browser.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
from collections.abc import Callable

import flet as ft
from local_engine import (
    MAX_IMAGE_BYTES,
    AnalysisError,
    analyze_manual_landmarks,
    analyze_mobile_front_points,
    analyze_mobile_side_photo,
    analyze_photo,
    image_aspect_ratio,
)

from front_landmarks import FRONT_LANDMARK_DEFS
from side_landmarks import SIDE_LANDMARK_DEFS

# Keep the design tokens aligned with the original HTML workspace.
BG = "#0F1118"
PANEL = "#171A23"
RAISED = "#232731"
SURFACE = "#12151D"
LINE = "#343946"
TEXT = "#E9E8F1"
MUTED = "#ACADBB"
CYAN = "#A0E9DC"
CYAN_DARK = "#003731"
BLUE = "#B9C6FF"
AMBER = "#F2D48A"
GREEN = "#B7E3BE"
RED = "#FFB4AB"

PROFILE_NAMES = {"front": "Front profile", "side": "Side profile"}
TABS = (
    ("photos", "01", "Photos", "Pair the views", ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED),
    ("align", "02", "Align", "Review the map", ft.Icons.GRID_VIEW_ROUNDED),
    ("results", "03", "Results", "Geometry", ft.Icons.ANALYTICS_OUTLINED),
    ("features", "04", "AI signal", "Independent model", ft.Icons.BIOTECH_ROUNDED),
)
ETHNICITY_OPTIONS = (
    ("asian", "Asian"),
    ("caucasian", "Caucasian"),
    ("black", "Black"),
    ("hispanic", "Hispanic"),
    ("middle_eastern", "Middle Eastern"),
    ("south_asian", "South Asian"),
    ("mixed", "Mixed"),
)


def _edge() -> ft.Border:
    side = ft.BorderSide(1, LINE)
    return ft.Border(top=side, right=side, bottom=side, left=side)


def _text(
    value: str,
    size: int = 13,
    color: str = TEXT,
    *,
    bold: bool = False,
    selectable: bool = False,
) -> ft.Text:
    # Most UI labels have no reason to create Flutter SelectableText objects.
    return ft.Text(
        value,
        size=size,
        color=color,
        weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
        selectable=selectable,
    )


def _eyebrow(value: str) -> ft.Text:
    return ft.Text(value.upper(), size=10, color=CYAN, weight=ft.FontWeight.BOLD)


def _card(
    *controls: ft.Control, bgcolor: str = PANEL, padding: int = 18
) -> ft.Container:
    return ft.Container(
        content=ft.Column(controls=list(controls), spacing=12),
        padding=padding,
        bgcolor=bgcolor,
        border=_edge(),
        border_radius=24,
    )


def _tag(value: str, *, color: str = CYAN) -> ft.Container:
    return ft.Container(
        content=_text(value, 11, color, bold=True),
        padding=ft.Padding(left=11, top=6, right=11, bottom=6),
        bgcolor=SURFACE,
        border_radius=18,
        border=_edge(),
    )


def _button(
    value: str,
    *,
    icon: ft.IconData | None = None,
    action: ft.ClientAction | None = None,
    on_click: Callable | None = None,
    primary: bool = False,
    disabled: bool = False,
    expand: bool = False,
) -> ft.Button:
    return ft.Button(
        content=value,
        icon=icon,
        bgcolor=CYAN if primary else RAISED,
        color=CYAN_DARK if primary else TEXT,
        elevation=0,
        action=action,
        on_click=on_click,
        disabled=disabled,
        expand=expand,
        height=50,
    )


def _score(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(10.0, float(value)))
    return None


def _display(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value) if value is not None else "—"


def _score_bar(value: object, label: str) -> ft.Control:
    score = _score(value)
    return ft.Column(
        spacing=5,
        controls=[
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    _text(label, 12, MUTED),
                    _text(
                        f"{score:.2f} / 10" if score is not None else "—",
                        12,
                        TEXT,
                        bold=True,
                    ),
                ],
            ),
            ft.ProgressBar(
                value=score / 10 if score is not None else 0,
                color=CYAN,
                bgcolor=LINE,
                bar_height=5,
                border_radius=5,
            ),
        ],
    )


def _image_preview(data: bytes, max_size: int = 960) -> bytes:
    """Make a bounded UI-only JPEG once on upload; analysis keeps original bytes.

    JPEG draft decoding is particularly useful for large phone-camera photos.
    Never resize the source used for landmark measurements.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as image:
            image.draft("RGB", (max_size, max_size))
            display = ImageOps.exif_transpose(image)
            display.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
            if display.mode != "RGB":
                rgba = display.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                display = background
            with io.BytesIO() as output:
                display.save(output, "JPEG", quality=78)
                return output.getvalue()
    except (OSError, ValueError):
        # Preserve existing display behavior for malformed images; inference
        # still validates the original, untouched user input.
        return data


def _photo_stage(
    photo: tuple[str, bytes] | None,
    title: str,
    *,
    large: bool = True,
    preview: bytes | None = None,
) -> ft.Control:
    if photo is None:
        content: ft.Control = ft.Column(
            controls=[
                ft.Icon(ft.Icons.FACE_RETOUCHING_NATURAL_ROUNDED, color=CYAN, size=44),
                _text("Photo workspace", 19, TEXT, bold=True),
                _text("Choose a portrait to preview it here.", 12, MUTED),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        )
    else:
        content = ft.Column(
            controls=[
                ft.Image(
                    src=preview if preview is not None else photo[1],
                    fit=ft.BoxFit.CONTAIN,
                    # Let Flutter preserve aspect ratio while decoding a small
                    # already-downscaled thumbnail rather than the source photo.
                    cache_width=768 if large else 480,
                    height=385 if large else 205,
                    border_radius=10,
                ),
                _text(photo[0], 11, MUTED),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )
    return ft.Container(
        content=content,
        alignment=ft.Alignment.CENTER,
        padding=20,
        height=490 if large else 295,
        bgcolor=SURFACE,
        border=_edge(),
        border_radius=14,
    )


def _measurement_cards(data: dict, profile: str, order: str) -> list[ft.Control]:
    key = "frontMeasurements" if profile == "front" else "sideMeasurements"
    source = data.get(key, [])
    if not isinstance(source, list):
        return []
    measurements = [entry for entry in source if isinstance(entry, dict)]
    if order != "importance":
        measurements.sort(
            key=lambda entry: _score(entry.get("score")) or 0,
            reverse=order == "high",
        )

    cards: list[ft.Control] = []
    for entry in measurements:
        title = str(entry.get("name", "Measurement"))
        value = _display(entry.get("value"))
        unit = str(entry.get("unit") or "")
        category = str(entry.get("category") or "Geometry")
        cards.append(
            _card(
                ft.Row(
                    controls=[
                        ft.Column(
                            controls=[
                                _text(title, 13, TEXT, bold=True),
                                _text(category, 10, MUTED),
                            ],
                            spacing=3,
                            expand=True,
                        ),
                        ft.Column(
                            controls=[
                                _text(f"{value} {unit}".strip(), 14, TEXT, bold=True),
                                _text(
                                    f"{_score(entry.get('score')):.1f} / 10"
                                    if _score(entry.get("score")) is not None
                                    else "—",
                                    11,
                                    CYAN,
                                ),
                            ],
                            spacing=3,
                            horizontal_alignment=ft.CrossAxisAlignment.END,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                _score_bar(entry.get("score"), "Metric score"),
                bgcolor=RAISED,
                padding=14,
            )
        )
    return cards


def _report_body(response: dict, profile: str, order: str) -> list[ft.Control]:
    data = response.get("data")
    if not isinstance(data, dict):
        missing = response.get("missing_ids", [])
        return [
            _card(
                _eyebrow("Incomplete landmark map"),
                _text(
                    str(response.get("warning") or "Analysis could not be completed."),
                    17,
                    AMBER,
                    bold=True,
                ),
                _text(
                    f"Detected landmarks: {response.get('landmark_count', 0)}",
                    12,
                    MUTED,
                ),
                _text(
                    "Missing points: " + ", ".join(str(item) for item in missing)
                    if isinstance(missing, list) and missing
                    else "Try a well-lit portrait with the full face outline visible.",
                    12,
                    MUTED,
                ),
                _text(
                    "This desktop version uses automatic detection. For manual point editing, "
                    "use the browser landmark canvas.",
                    12,
                    MUTED,
                ),
            )
        ]

    score = _score(data.get("overallScore"))
    head = _card(
        _eyebrow("Geometry report"),
        ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Column(
                    controls=[
                        _text("Overall analysis", 13, MUTED),
                        _text(
                            f"{score:.2f}" if score is not None else "—",
                            42,
                            TEXT,
                            bold=True,
                        ),
                        _text("Model-specific scale · 0–10", 11, MUTED),
                    ],
                    spacing=3,
                ),
                _tag(f"{response.get('landmark_count', 0)} landmarks", color=GREEN),
            ],
        ),
        _score_bar(data.get("overallScore"), "Overall score"),
        _score_bar(
            data.get("frontScore") if profile == "front" else data.get("sideScore"),
            PROFILE_NAMES[profile],
        ),
        bgcolor=RAISED,
    )
    content: list[ft.Control] = [head]

    categories = data.get("categoryScores")
    if isinstance(categories, dict) and categories:
        content.append(
            _card(
                _eyebrow("Category breakdown"),
                *[
                    _score_bar(value, str(key).replace("_", " ").title())
                    for key, value in categories.items()
                ],
            )
        )

    for key, title, color in (
        ("topStrengths", "Higher-scoring metrics", GREEN),
        ("topWeaknesses", "Lower-scoring metrics", AMBER),
    ):
        values = data.get(key)
        if isinstance(values, list) and values:
            content.append(
                _card(
                    _text(title, 14, color, bold=True),
                    *[_text(f"• {item}", 12) for item in values[:3]],
                )
            )

    measurements = _measurement_cards(data, profile, order)
    content.append(
        _text(f"All measurements  ·  {len(measurements)}", 18, TEXT, bold=True)
    )
    content.extend(
        measurements
        if measurements
        else [_text("No measurement rows returned.", 12, MUTED)]
    )
    return content


def _feature_body(response: dict | None) -> list[ft.Control]:
    if response is None:
        return [
            _card(
                ft.Icon(ft.Icons.BIOTECH_ROUNDED, size=35, color=BLUE),
                _text("No feature analysis yet", 20, TEXT, bold=True),
                _text(
                    "Upload a clear front view, then run the optional AI model.",
                    12,
                    MUTED,
                ),
            )
        ]
    data = response.get("data")
    if not isinstance(data, dict):
        return [_card(_text("No feature-rating data returned.", 13, AMBER))]

    content: list[ft.Control] = [
        _card(
            _eyebrow("Independent learned signal"),
            _text(f"{_display(data.get('score10'))} / 10", 37, TEXT, bold=True),
            _text(f"Raw model score: {_display(data.get('rawScore'))}", 12, MUTED),
            _text("This output is not included in geometric measurements.", 11, MUTED),
            bgcolor=RAISED,
        )
    ]
    heatmap = data.get("heatmapPng")
    if isinstance(heatmap, str) and heatmap:
        try:
            decoded = base64.b64decode(heatmap.partition(",")[2], validate=True)
        except (ValueError, binascii.Error):
            content.append(_text("Heatmap preview unavailable.", 12, AMBER))
        else:
            content.append(
                _card(
                    _eyebrow("Occlusion heatmap"),
                    ft.Image(src=decoded, fit=ft.BoxFit.CONTAIN, height=305),
                    _text(
                        "Where blurring a region changed the model prediction.",
                        11,
                        MUTED,
                    ),
                )
            )

    regions = data.get("regions")
    if isinstance(regions, list) and regions:
        cards: list[ft.Control] = [_text("Regional contribution", 18, TEXT, bold=True)]
        for region in regions:
            if not isinstance(region, dict):
                continue
            delta = region.get("delta")
            cards.append(
                _card(
                    ft.Row(
                        controls=[
                            _text(
                                str(region.get("name") or "Region"), 13, TEXT, bold=True
                            ),
                            _tag(
                                f"{float(delta):+.4f}"
                                if isinstance(delta, (int, float))
                                else "—"
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    _text(str(region.get("effect") or ""), 11, MUTED),
                    bgcolor=RAISED,
                    padding=13,
                )
            )
        content.extend(cards)
    summary = data.get("summary")
    if isinstance(summary, str) and summary.strip():
        content.append(
            _card(_eyebrow("Model notes"), _text(summary, 12, MUTED, selectable=True))
        )
    return content


def main(page: ft.Page) -> None:
    page.title = "mini-faceIQ · Facial intelligence lab"
    page.bgcolor = BG
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=CYAN, use_material3=True)
    page.dark_theme = ft.Theme(color_scheme_seed=CYAN, use_material3=True)
    page.adaptive = False  # Always render Android-like Material, including on iOS.
    page.padding = 0
    # Desktop window sizing must not override Android/iOS device dimensions.
    if page.platform in (
        ft.PagePlatform.LINUX,
        ft.PagePlatform.WINDOWS,
        ft.PagePlatform.MACOS,
    ):
        page.window.width = 1080
        page.window.height = 620
        page.window.min_width = 340
        page.window.min_height = 480

    photos: dict[str, tuple[str, bytes]] = {}
    responses: dict[str, dict] = {}
    feature_photo: tuple[str, bytes] | None = None
    # Display-only compressed previews. Original bytes stay in photos for ML.
    previews: dict[str, bytes] = {}
    active_tab = "photos"
    active_profile = "front"
    metric_order = "importance"
    busy = False
    android_shell = page.platform == ft.PagePlatform.ANDROID
    ios_shell = page.platform == ft.PagePlatform.IOS
    native_face_shell = android_shell or ios_shell
    face_service = None
    if native_face_shell:
        # Both mobile platforms use their native MediaPipe Tasks bridge.
        # Keep desktop/FakePage tests importable without the extension.
        try:
            import importlib

            extension = importlib.import_module("flet_face_landmarker")
            face_service = extension.FletFaceLandmarker()
            page.services.append(face_service)
        except ImportError:
            face_service = None
    manual_points: dict[str, dict[str, dict]] = {"front": {}, "side": {}}
    manual_selected: dict[str, int] = {"front": 0, "side": 0}
    manual_history: dict[str, list[dict[str, dict]]] = {"front": [], "side": []}
    status = (
        "Android: MediaPipe + 3DDFA + SCUT run offline on-device."
        if android_shell
        else (
            "iPhone: offline front MediaPipe and manual front/side geometry. "
            "Side auto-detection and SCUT are not yet available on iOS."
            if ios_shell
            else "Ready for local analysis · no API or uploads"
        )
    )
    status_color = MUTED
    # Keep status controls stable so progress updates do not rebuild large
    # image previews, landmark editors, or measurement cards on Android.
    status_label = _text(status, 12, status_color)
    status_progress = ft.ProgressBar(color=CYAN, bgcolor=LINE, visible=False)
    status_idle = _tag("On-device Python", color=GREEN)

    gender = ft.Dropdown(
        label="Gender reference",
        value="male",
        options=[
            ft.DropdownOption(key="male", text="Male"),
            ft.DropdownOption(key="female", text="Female"),
        ],
        expand=True,
        col={"xs": 12, "sm": 6},
        bgcolor=RAISED,
    )
    ethnicity = ft.Dropdown(
        label="Ethnicity reference",
        value="asian",
        options=[
            ft.DropdownOption(key=key, text=label) for key, label in ETHNICITY_OPTIONS
        ],
        expand=True,
        col={"xs": 12, "sm": 6},
        bgcolor=RAISED,
    )

    # Material 3 navigation: bottom bar for phones, rail + workspace for wide windows.
    # Six persistent views. Flutter receives visibility changes on navigation,
    # not hundreds of brand-new controls and raw photo bytes on every tap.
    view_keys: tuple[tuple[str, str | None], ...] = (
        ("photos", None),
        ("align", "front"),
        ("align", "side"),
        ("results", "front"),
        ("results", "side"),
        ("features", None),
    )
    sidebar_views: dict[tuple[str, str | None], ft.Container] = {
        key: ft.Container(content=ft.Column(spacing=18), visible=False)
        for key in view_keys
    }
    workspace_views: dict[tuple[str, str | None], ft.Container] = {
        key: ft.Container(content=ft.Column(spacing=18), visible=False)
        for key in view_keys
    }
    view_signatures: dict[tuple[str, str | None], tuple] = {}
    last_view: tuple[str, str | None] | None = None
    sidebar_content = ft.Column(controls=list(sidebar_views.values()), spacing=0)
    workspace_content = ft.Column(controls=list(workspace_views.values()), spacing=0)
    root = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
    last_compact: bool | None = None

    def compact_screen() -> bool:
        return (page.width or 1100) < 840

    def navigate_index(index: int | None) -> None:
        if index is not None and 0 <= index < len(TABS):
            switch_tab(TABS[index][0])

    bottom_navigation = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=icon, selected_icon=icon, label=title)
            for _tab, _index, title, _subtitle, icon in TABS
        ],
        selected_index=0,
        bgcolor=PANEL,
        indicator_color=RAISED,
        label_behavior=ft.NavigationBarLabelBehavior.ALWAYS_SHOW,
        on_change=lambda event: navigate_index(event.control.selected_index),
    )
    rail_navigation = ft.NavigationRail(
        destinations=[
            ft.NavigationRailDestination(icon=icon, selected_icon=icon, label=title)
            for _tab, _index, title, _subtitle, icon in TABS
        ],
        selected_index=0,
        bgcolor=PANEL,
        on_change=lambda event: navigate_index(event.control.selected_index),
    )

    rail_host = ft.Container(
        content=rail_navigation,
        width=92,
        height=max(390, (page.height or 680) - 112),
        padding=ft.Padding(left=4, top=10, right=4, bottom=8),
        bgcolor=PANEL,
        border_radius=24,
    )

    def reflow() -> None:
        nonlocal last_compact
        # NavigationRail requires a bounded height, even inside scrollable layouts.
        rail_host.height = max(390, (page.height or 680) - 112)
        mobile = compact_screen()
        if mobile == last_compact:
            return
        last_compact = mobile
        if mobile:
            root.controls = [
                ft.Container(content=sidebar_content, padding=14),
                ft.Container(content=workspace_content, padding=14),
            ]
            page.navigation_bar = bottom_navigation
        else:
            root.controls = [
                ft.Row(
                    controls=[
                        rail_host,
                        ft.Container(
                            content=sidebar_content,
                            width=350,
                            padding=20,
                            bgcolor=PANEL,
                            border_radius=24,
                        ),
                        ft.Container(
                            content=workspace_content,
                            expand=True,
                            padding=ft.Padding(left=12, top=10, right=16, bottom=14),
                        ),
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ]
            page.navigation_bar = None

    def set_status(message: str, *, color: str = MUTED) -> None:
        nonlocal status, status_color
        status = message
        status_color = color

    def sync_status_controls() -> None:
        status_label.value = status
        status_label.color = status_color
        status_progress.visible = busy
        status_idle.visible = not busy

    def refresh_status_only() -> None:
        """Patch three small controls; don't traverse the entire page."""
        sync_status_controls()
        page.update(status_label, status_progress, status_idle)

    def switch_tab(tab: str) -> None:
        nonlocal active_tab
        if tab == active_tab:
            return
        active_tab = tab
        render()

    def switch_profile(profile: str) -> None:
        nonlocal active_profile
        if profile == active_profile:
            return
        active_profile = profile
        render()

    def change_order(event: ft.Event[ft.Dropdown]) -> None:
        nonlocal metric_order
        metric_order = event.control.value or "importance"
        render()

    def change_context(_event: ft.Event[ft.Dropdown]) -> None:
        responses.clear()
        set_status("Reference changed. Re-run geometry to update scores.", color=AMBER)
        render()

    gender.on_select = change_context
    ethnicity.on_select = change_context

    def store_photo(profile: str, name: str, contents: bytes) -> None:
        nonlocal feature_photo, active_profile
        image = (name, contents)
        # Create a display-only preview once; keep original pixels for ML.
        previews[profile] = _image_preview(contents)
        if profile == "features":
            feature_photo = image
            responses.pop("features", None)
        else:
            photos[profile] = image
            manual_points[profile].clear()
            manual_history[profile].clear()
            manual_selected[profile] = 0
            responses.pop(profile, None)
            if profile == "front" and feature_photo is None:
                responses.pop("features", None)
            active_profile = profile
        set_status(f"{name} loaded · ready to analyze", color=GREEN)
        render()

    def picked(profile: str, event: ft.FilePickerResultEvent) -> None:
        if not event.files:
            return
        file = event.files[0]
        if not file.bytes:
            set_status("The file picker could not read this photo.", color=AMBER)
        elif len(file.bytes) > MAX_IMAGE_BYTES:
            set_status("Photo exceeds the 20 MB local limit.", color=AMBER)
        elif ios_shell and file.name.lower().endswith((".heic", ".heif")):
            if face_service is None:
                set_status(
                    "HEIC needs the iOS native extension. Use a JPEG or PNG instead.",
                    color=AMBER,
                )
            else:
                native_service = face_service

                async def convert_heic() -> None:
                    try:
                        jpeg = await native_service.convert_to_jpeg(file.bytes)
                        if not jpeg or len(jpeg) > MAX_IMAGE_BYTES:
                            raise ValueError("Converted photo exceeds the 20 MB limit.")
                        store_photo(profile, file.name.rsplit(".", 1)[0] + ".jpg", jpeg)
                    except Exception as exc:  # noqa: BLE001 - native plugin errors vary.
                        set_status(
                            f"iOS HEIC conversion failed: {type(exc).__name__}: {str(exc)[:140]}",
                            color=RED,
                        )
                        render()

                set_status("Converting HEIC to JPEG offline on this iPhone...", color=CYAN)
                page.run_task(convert_heic)
        elif not file.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            set_status("Use JPG, PNG, WebP, or iPhone HEIC photos.", color=AMBER)
        else:
            store_photo(profile, file.name, file.bytes)
            return
        render()

    pickers = {
        profile: ft.FilePicker(on_result=lambda event, p=profile: picked(p, event))
        for profile in ("front", "side", "features")
    }
    page.services.extend(pickers.values())

    def pick_action(profile: str) -> ft.PickFiles:
        return ft.PickFiles(
            pickers[profile],
            dialog_title="Select a portrait",
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["jpg", "jpeg", "png", "webp"]
            + (["heic", "heif"] if ios_shell else []),
            with_data=True,
        )

    async def compute_manual(profile: str) -> None:
        """Offline front/side geometry path; no native detector packages required."""
        nonlocal busy, active_tab, active_profile
        photo = photos.get(profile)
        if photo is None:
            set_status(f"Choose a {profile} portrait on Photos first.", color=AMBER)
            active_tab = "photos"
            render()
            return
        if busy:
            return
        busy = True
        set_status(
            f"Calculating {profile} landmarks using the desktop formulas...", color=CYAN
        )
        render()
        try:
            result = await asyncio.to_thread(
                analyze_manual_landmarks,
                profile,
                photo[1],
                list(manual_points[profile].values()),
                gender.value or "male",
                ethnicity.value or "asian",
            )
            responses[profile] = result
            active_profile = profile
            if result.get("data") is None:
                active_tab = "align"
                set_status(
                    result.get("warning", "Place all required points."), color=AMBER
                )
            else:
                active_tab = "results"
                set_status(
                    f"{profile.title()} manual geometry completed offline.", color=GREEN
                )
        except (
            AnalysisError,
            ImportError,
            OSError,
            ArithmeticError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            set_status(
                f"Manual analysis failed ({type(exc).__name__}): {str(exc)[:240]}",
                color=RED,
            )
        finally:
            busy = False
            render()

    async def compute_mobile_front() -> None:
        """Run the SAME MediaPipe .task with the original front point index map."""
        nonlocal busy, active_tab, active_profile
        photo = photos.get("front")
        if photo is None:
            set_status("Choose a front portrait first.", color=AMBER)
            active_tab = "photos"
            render()
            return
        if busy:
            return
        if face_service is None:
            set_status(
                "Native MediaPipe extension is missing; use Align for manual points.",
                color=AMBER,
            )
            active_tab = "align"
            active_profile = "front"
            render()
            return
        busy = True
        set_status(
            "Running original MediaPipe face landmarker on-device...", color=CYAN
        )
        refresh_status_only()
        try:
            from mobile_front import get_front_task_model

            normalized = await face_service.detect(photo[1], get_front_task_model())
            result = await asyncio.to_thread(
                analyze_mobile_front_points,
                photo[1],
                normalized,
                gender.value or "male",
                ethnicity.value or "asian",
            )
            responses["front"] = result
            manual_points["front"] = {
                point["id"]: point for point in result.get("landmarks", [])
            }
            active_profile = "front"
            active_tab = "results" if result.get("data") is not None else "align"
            set_status(
                "Front MediaPipe analysis complete."
                if result.get("data") is not None
                else result.get("warning", "Some front landmarks need correction."),
                color=GREEN if result.get("data") is not None else AMBER,
            )
        except Exception as exc:  # noqa: BLE001 - native plugin reports platform failures
            active_tab = "align"
            active_profile = "front"
            set_status(
                f"Native MediaPipe failed ({type(exc).__name__}): {str(exc)[:180]}. "
                "Manual landmark placement is still available.",
                color=RED,
            )
        finally:
            busy = False
            render()

    async def compute_mobile_features() -> None:
        """Exact SCUT checkpoint exported to ONNX, with eight occlusion passes."""
        nonlocal busy, active_tab
        photo = feature_photo or photos.get("front")
        if photo is None:
            set_status("Choose a feature photo first.", color=AMBER)
            active_tab = "features"
            render()
            return
        if busy:
            return
        busy = True
        active_tab = "features"
        set_status(
            "Running SCUT ONNX rating and eight region occlusions...", color=CYAN
        )
        refresh_status_only()
        try:
            from mobile_features import analyze_mobile_features, resized_face_photo
            from mobile_front import get_front_task_model

            normalized = (
                await face_service.detect(
                    resized_face_photo(photo[1]), get_front_task_model()
                )
                if face_service is not None
                else []
            )
            result = await asyncio.to_thread(
                analyze_mobile_features, photo[1], normalized
            )
            responses["features"] = {
                "mode": "features",
                "data": result,
                "landmark_count": len(normalized),
            }
            set_status(
                "Original SCUT ONNX rating and heatmap completed offline.", color=GREEN
            )
        except Exception as exc:  # noqa: BLE001 - model/platform errors are shown to the user
            set_status(
                f"SCUT Android inference failed ({type(exc).__name__}): {str(exc)[:240]}",
                color=RED,
            )
        finally:
            busy = False
            render()

    async def compute_mobile_side() -> None:
        """Automatic, fully local Android side inference with the existing 3DDFA models."""
        nonlocal busy, active_tab, active_profile
        photo = photos.get("side")
        if photo is None:
            set_status("Choose a side portrait first.", color=AMBER)
            active_tab = "photos"
            render()
            return
        if busy:
            return
        busy = True
        set_status("Running original 3DDFA ONNX models on device...", color=CYAN)
        refresh_status_only()
        try:
            result = await asyncio.to_thread(
                analyze_mobile_side_photo,
                photo[0],
                photo[1],
                gender.value or "male",
                ethnicity.value or "asian",
            )
            responses["side"] = result
            active_profile = "side"
            if isinstance(result.get("landmarks"), list):
                manual_points["side"] = {
                    point["id"]: point for point in result["landmarks"]
                }
            active_tab = "results" if result.get("data") is not None else "align"
            set_status(
                "Side ONNX analysis complete."
                if result.get("data") is not None
                else result.get(
                    "warning", "Some side landmarks need manual placement."
                ),
                color=GREEN if result.get("data") is not None else AMBER,
            )
        except (
            AnalysisError,
            ImportError,
            OSError,
            ArithmeticError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            active_tab = "align"
            active_profile = "side"
            set_status(
                f"Side ONNX unavailable ({type(exc).__name__}): {str(exc)[:180]}. "
                "You can still place landmarks manually.",
                color=RED,
            )
        finally:
            busy = False
            render()

    async def run_analysis(profile: str) -> None:
        nonlocal busy, active_tab, active_profile
        if ios_shell and profile == "side":
            # The Python ONNX/OpenCV side engine has Android, not iOS, wheels.
            # Never fall through to the Linux-only PyTorch detector on iPhone.
            await compute_manual("side")
            return
        if android_shell and profile == "side":
            await compute_mobile_side()
            return
        if native_face_shell and profile == "front":
            await compute_mobile_front()
            return
        if ios_shell and profile == "features":
            set_status(
                "SCUT inference is not yet ported to native iOS ONNX Runtime.",
                color=AMBER,
            )
            active_tab = "features"
            render()
            return
        if android_shell and profile == "features":
            await compute_mobile_features()
            return
        image = (
            feature_photo or photos.get("front")
            if profile == "features"
            else photos.get(profile)
        )
        if image is None:
            set_status(
                f"Choose a {'front' if profile == 'features' else profile} photo first.",
                color=AMBER,
            )
            render()
            return
        if busy:
            return

        busy = True
        set_status(f"Running {profile} analysis locally...", color=CYAN)
        refresh_status_only()
        try:
            result = await asyncio.to_thread(
                analyze_photo,
                profile,
                image[0],
                image[1],
                gender.value or "male",
                ethnicity.value or "asian",
            )
            responses[profile] = result
            if profile == "features":
                active_tab = "features"
            else:
                active_profile = profile
                active_tab = "results" if result.get("data") is not None else "align"
            complete = result.get("data") is not None
            set_status(
                f"{PROFILE_NAMES.get(profile, 'AI feature')} analysis finished."
                if complete
                else "Partial map: required landmarks are missing.",
                color=GREEN if complete else AMBER,
            )
        except AnalysisError as exc:
            set_status(str(exc), color=AMBER)
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            # Handle common failures from native inference and its optional packages.
            set_status(
                f"Local inference failed ({type(exc).__name__}): {str(exc)[:280]}",
                color=RED,
            )
        finally:
            busy = False
            render()

    def analyze_action(profile: str) -> Callable:
        async def handler(_event: ft.Event[ft.Button]) -> None:
            await run_analysis(profile)

        return handler

    async def run_pair(_event: ft.Event[ft.Button]) -> None:
        nonlocal busy, active_tab, active_profile
        if native_face_shell:
            targets = [key for key in ("front", "side") if key in photos]
            if not targets:
                set_status("Select a portrait first, then open Align.", color=AMBER)
                render()
                return
            if busy:
                return
            busy = True
            failures: list[str] = []
            try:
                for profile in targets:
                    photo = photos[profile]
                    set_status(f"Analyzing {profile} on-device...", color=CYAN)
                    refresh_status_only()
                    try:
                        if profile == "side" and android_shell:
                            result = await asyncio.to_thread(
                                analyze_mobile_side_photo,
                                photo[0],
                                photo[1],
                                gender.value or "male",
                                ethnicity.value or "asian",
                            )
                        elif profile == "front" and face_service is not None:
                            from mobile_front import get_front_task_model

                            normalized = await face_service.detect(
                                photo[1], get_front_task_model()
                            )
                            result = await asyncio.to_thread(
                                analyze_mobile_front_points,
                                photo[1],
                                normalized,
                                gender.value or "male",
                                ethnicity.value or "asian",
                            )
                        else:
                            result = await asyncio.to_thread(
                                analyze_manual_landmarks,
                                profile,
                                photo[1],
                                list(manual_points[profile].values()),
                                gender.value or "male",
                                ethnicity.value or "asian",
                            )
                        responses[profile] = result
                        if isinstance(result.get("landmarks"), list):
                            manual_points[profile] = {
                                point["id"]: point for point in result["landmarks"]
                            }
                    except Exception as exc:  # noqa: BLE001 - independent model failures
                        responses.pop(profile, None)
                        failures.append(
                            f"{profile}: {type(exc).__name__}: {str(exc)[:420]}"
                        )
                incomplete = next(
                    (
                        key
                        for key in targets
                        if responses.get(key, {}).get("data") is None
                    ),
                    None,
                )
                active_profile = incomplete or targets[0]
                active_tab = "align" if incomplete else "results"
                if failures:
                    set_status(
                        "Some automatic analyses failed: " + "; ".join(failures),
                        color=RED,
                    )
                elif incomplete:
                    set_status(
                        f"{incomplete.title()}: finish placing its landmarks in Align.",
                        color=AMBER,
                    )
                else:
                    set_status("All loaded profiles calculated offline.", color=GREEN)
            finally:
                busy = False
                render()
            return
        modes = [profile for profile in ("front", "side") if profile in photos]
        if not modes:
            set_status("Choose at least one photo first.", color=AMBER)
            render()
            return
        if busy:
            return
        busy = True
        for profile in modes:
            photo = photos[profile]
            set_status(f"Analyzing {profile} profile...", color=CYAN)
            refresh_status_only()
            try:
                result = await asyncio.to_thread(
                    analyze_photo,
                    profile,
                    photo[0],
                    photo[1],
                    gender.value or "male",
                    ethnicity.value or "asian",
                )
                responses[profile] = result
            except AnalysisError as exc:
                responses.pop(profile, None)
                set_status(f"{profile}: {exc}", color=AMBER)
            except (
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                responses.pop(profile, None)
                set_status(
                    f"{profile} failed ({type(exc).__name__}): {str(exc)[:200]}",
                    color=RED,
                )
        busy = False
        active_tab = "results"
        active_profile = modes[0]
        if all(
            profile in responses and responses[profile].get("data") is not None
            for profile in modes
        ):
            set_status("Local geometry analysis completed.", color=GREEN)
        render()

    def profile_tabs() -> ft.Row:
        return ft.Row(
            controls=[
                _button(
                    "Front view",
                    primary=active_profile == "front",
                    on_click=lambda _: switch_profile("front"),
                    expand=True,
                ),
                _button(
                    "Side view",
                    primary=active_profile == "side",
                    on_click=lambda _: switch_profile("side"),
                    expand=True,
                ),
            ],
            spacing=8,
        )

    def upload_field(profile: str) -> ft.Control:
        photo = photos.get(profile)
        return _card(
            ft.Row(
                controls=[
                    ft.Icon(
                        ft.Icons.FACE_OUTLINED
                        if profile == "front"
                        else ft.Icons.PORTRAIT_OUTLINED,
                        color=CYAN if photo else MUTED,
                    ),
                    ft.Column(
                        controls=[
                            _text(
                                f"{'Front-view' if profile == 'front' else 'Side-profile'} photo",
                                14,
                                TEXT,
                                bold=True,
                            ),
                            _text(
                                "Eyes level · neutral angle"
                                if profile == "front"
                                else "True 90° profile · full outline",
                                11,
                                MUTED,
                            ),
                        ],
                        spacing=4,
                        expand=True,
                    ),
                ],
            ),
            _text(
                f"{photo[0]}  ·  {len(photo[1]) / 1024:.0f} KB"
                if photo
                else "No photo selected",
                11,
                GREEN if photo else MUTED,
            ),
            _button(
                "Replace photo" if photo else "Choose photo",
                icon=ft.Icons.UPLOAD_FILE_ROUNDED,
                action=pick_action(profile),
                disabled=busy,
            ),
            bgcolor=RAISED,
        )

    def manual_editor() -> ft.Control:
        """Touch editor with normalized image coordinates, shared on desktop/Android."""
        profile = active_profile
        photo = photos.get(profile)
        if photo is None:
            return _card(
                _eyebrow("Manual landmarks"),
                _text(f"Select a {profile} photo in Photos first.", 13, MUTED),
                _button("Open photos", on_click=lambda _: switch_tab("photos")),
                bgcolor=RAISED,
            )
        definitions = FRONT_LANDMARK_DEFS if profile == "front" else SIDE_LANDMARK_DEFS
        points = manual_points[profile]
        index = min(manual_selected[profile], len(definitions) - 1)
        current_id = definitions[index]["id"]
        width = 300.0 if compact_screen() else 285.0
        # Preserve image aspect ratio: x/y positions refer to actual photo pixels.
        # Preview already matches EXIF orientation shown by the UI; measure
        # its dimensions so touch coordinates align on rotated phone photos.
        aspect = max(0.1, image_aspect_ratio(previews.get(profile, photo[1])))
        height = width / aspect
        marker_controls: dict[str, ft.Container] = {}
        active_drag: list[str | None] = [None]

        def xy(event) -> tuple[float, float]:
            location = event.local_position
            return (
                max(0.0, min(1.0, float(location.x) / width)),
                max(0.0, min(1.0, float(location.y) / height)),
            )

        def change_selection(event: ft.Event[ft.Dropdown]) -> None:
            chosen = event.control.value
            for position, definition in enumerate(definitions):
                if chosen == definition["id"]:
                    manual_selected[profile] = position
                    render()
                    return

        def place(event) -> None:
            x, y = xy(event)
            manual_history[profile].append(
                {key: value.copy() for key, value in points.items()}
            )
            points[current_id] = {"id": current_id, "x": x, "y": y}
            responses.pop(profile, None)
            # The next unset point is selected automatically.
            for position in range(index + 1, len(definitions)):
                if definitions[position]["id"] not in points:
                    manual_selected[profile] = position
                    break
            else:
                for position, definition in enumerate(definitions):
                    if definition["id"] not in points:
                        manual_selected[profile] = position
                        break
            set_status(
                f"{profile.title()}: {len(points)}/{len(definitions)} points placed.",
                color=GREEN,
            )
            render()

        def pan_start(event) -> None:
            x, y = xy(event)
            if not points:
                return
            nearest = min(
                points.values(),
                key=lambda item: (item["x"] - x) ** 2 + (item["y"] - y) ** 2,
            )
            if ((nearest["x"] - x) ** 2 + (nearest["y"] - y) ** 2) ** 0.5 > 0.075:
                return
            active_drag[0] = nearest["id"]
            manual_history[profile].append(
                {key: value.copy() for key, value in points.items()}
            )

        def pan_update(event) -> None:
            point_id = active_drag[0]
            if point_id is None:
                return
            x, y = xy(event)
            points[point_id] = {"id": point_id, "x": x, "y": y}
            responses.pop(profile, None)
            marker = marker_controls.get(point_id)
            if marker is not None:
                marker.left = x * width - 9
                marker.top = y * height - 9
                marker.update()

        def pan_end(_event) -> None:
            if active_drag[0] is not None:
                active_drag[0] = None
                render()

        def undo(_event) -> None:
            if manual_history[profile]:
                manual_points[profile] = manual_history[profile].pop()
                responses.pop(profile, None)
                render()

        async def calculate(_event) -> None:
            await compute_manual(profile)

        markers: list[ft.Control] = [
            ft.Image(
                src=previews.get(profile, photo[1]),
                width=width,
                height=height,
                fit=ft.BoxFit.FILL,
                cache_width=768,
            )
        ]
        for position, definition in enumerate(definitions):
            point = points.get(definition["id"])
            if point is None:
                continue
            marker = ft.Container(
                left=point["x"] * width - 9,
                top=point["y"] * height - 9,
                width=18,
                height=18,
                border_radius=9,
                bgcolor=CYAN if definition["id"] == current_id else definition["color"],
                alignment=ft.Alignment.CENTER,
                content=_text(str(position + 1), 8, CYAN_DARK, bold=True),
            )
            markers.append(marker)
            marker_controls[definition["id"]] = marker

        editor = ft.GestureDetector(
            content=ft.Stack(controls=markers, width=width, height=height),
            on_tap_down=place,
            on_pan_start=pan_start,
            on_pan_update=pan_update,
            on_pan_end=pan_end,
            drag_interval=40,
        )
        return _card(
            _eyebrow("Manual point editor · offline"),
            _text(
                f"{len(points)} / {len(definitions)} landmarks placed",
                13,
                GREEN,
                bold=True,
            ),
            _text(
                "Tap the next point on the portrait. Drag an existing dot to correct it.",
                11,
                MUTED,
            ),
            ft.Dropdown(
                label="Point to place or reposition",
                value=current_id,
                options=[
                    ft.DropdownOption(
                        key=definition["id"], text=f"{i + 1}. {definition['label']}"
                    )
                    for i, definition in enumerate(definitions)
                ],
                on_select=change_selection,
                bgcolor=SURFACE,
            ),
            ft.Container(
                content=editor,
                width=width,
                height=height,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
                border_radius=14,
            ),
            ft.Row(
                controls=[
                    _button(
                        "Undo",
                        icon=ft.Icons.UNDO_ROUNDED,
                        on_click=undo,
                        disabled=not manual_history[profile],
                        expand=True,
                    ),
                    _button(
                        "Calculate",
                        icon=ft.Icons.ANALYTICS_OUTLINED,
                        on_click=calculate,
                        disabled=busy or len(points) < len(definitions),
                        primary=True,
                        expand=True,
                    ),
                ],
                spacing=8,
            ),
            _text(
                "Uses the same Python front/side calculator functions as Linux. "
                "Automatic detectors and the AI rating are separate ports.",
                10,
                MUTED,
            ),
            bgcolor=RAISED,
            padding=12,
        )

    def sidebar() -> list[ft.Control]:
        # The compact AppBar already carries the brand. Avoid duplicating it
        # above the actual photo controls on an Android-sized viewport.
        body: list[ft.Control] = []
        if not compact_screen():
            body.extend(
                [
                    ft.Row(
                        controls=[
                            ft.Container(
                                content=_text("FQ", 20, CYAN_DARK, bold=True),
                                alignment=ft.Alignment.CENTER,
                                bgcolor=CYAN,
                                width=51,
                                height=51,
                                border_radius=15,
                            ),
                            ft.Column(
                                controls=[
                                    _eyebrow("Facial intelligence lab"),
                                    _text("mini-faceIQ", 25, TEXT, bold=True),
                                ],
                                spacing=3,
                                expand=True,
                            ),
                        ],
                        spacing=12,
                    ),
                    _card(
                        _eyebrow("Current workspace"),
                        ft.Row(
                            controls=[
                                _text(
                                    PROFILE_NAMES[active_profile], 14, TEXT, bold=True
                                ),
                                _tag("Active"),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        bgcolor=RAISED,
                        padding=13,
                    ),
                ]
            )

        if active_tab == "photos":
            body.extend(
                [
                    _eyebrow("Step 01 / Input"),
                    _text("Build the analysis set", 20, TEXT, bold=True),
                    _text(
                        "Pair a straight-on portrait with a clean side profile.",
                        12,
                        MUTED,
                    ),
                    upload_field("front"),
                    upload_field("side"),
                    _card(
                        _eyebrow("Scoring context"),
                        ft.ResponsiveRow(
                            controls=[gender, ethnicity], spacing=8, run_spacing=8
                        ),
                        bgcolor=RAISED,
                        padding=12,
                    ),
                    _button(
                        "Auto-analyze front + side"
                        if android_shell
                        else (
                            "Detect front · manually align side"
                            if ios_shell
                            else "Analyze available photos"
                        ),
                        icon=ft.Icons.AUTO_AWESOME_ROUNDED,
                        on_click=run_pair,
                        primary=True,
                        disabled=busy or not photos,
                    ),
                    _text(
                        "Each profile runs independently; the missing view is never invented.",
                        11,
                        MUTED,
                    ),
                ]
            )
        elif active_tab == "align":
            report = responses.get(active_profile)
            body.extend(
                [
                    _eyebrow("Step 02 / Geometry"),
                    _text("Review landmark readiness", 20, TEXT, bold=True),
                    _text(
                        "Tap and adjust every required point to compute the real geometry offline.",
                        12,
                        MUTED,
                    ),
                    profile_tabs(),
                    _card(
                        _eyebrow("Detection status"),
                        _text(
                            f"{report.get('landmark_count', 0)} points detected"
                            if report
                            else "No detection yet",
                            17,
                            TEXT,
                            bold=True,
                        ),
                        _text(
                            "Map complete"
                            if report and report.get("data") is not None
                            else "Awaiting complete map",
                            11,
                            GREEN
                            if report and report.get("data") is not None
                            else AMBER,
                        ),
                        bgcolor=RAISED,
                    ),
                    _button(
                        "Auto-detect front/side"
                        if android_shell
                        else (
                            "Detect front automatically"
                            if ios_shell and active_profile == "front"
                            else (
                                "Calculate manual geometry"
                                if ios_shell
                                else "Auto-detect + analyze"
                            )
                        ),
                        icon=ft.Icons.AUTO_AWESOME_ROUNDED,
                        on_click=analyze_action(active_profile),
                        primary=True,
                        disabled=busy or active_profile not in photos,
                    ),
                    manual_editor(),
                ]
            )
        elif active_tab == "results":
            body.extend(
                [
                    _eyebrow("Step 03 / Results"),
                    _text("Read the geometry", 20, TEXT, bold=True),
                    _text(
                        "Headline scores, categories, and every measured ratio.",
                        12,
                        MUTED,
                    ),
                    profile_tabs(),
                    _button(
                        "Refresh current profile",
                        icon=ft.Icons.REFRESH_ROUNDED,
                        on_click=analyze_action(active_profile),
                        primary=True,
                        disabled=busy or active_profile not in photos,
                    ),
                    _button(
                        "Refresh both profiles",
                        icon=ft.Icons.AUTO_AWESOME_ROUNDED,
                        on_click=run_pair,
                        disabled=busy or not photos,
                    ),
                    ft.Dropdown(
                        label="Sort measurements",
                        value=metric_order,
                        options=[
                            ft.DropdownOption(key="importance", text="Default order"),
                            ft.DropdownOption(key="high", text="Highest score first"),
                            ft.DropdownOption(key="low", text="Lowest score first"),
                        ],
                        on_select=change_order,
                        bgcolor=RAISED,
                    ),
                    _button(
                        "View AI feature signal",
                        icon=ft.Icons.BIOTECH_ROUNDED,
                        on_click=lambda _: switch_tab("features"),
                    ),
                ]
            )
        else:
            feature = feature_photo or photos.get("front")
            body.extend(
                [
                    _eyebrow("Independent / AI"),
                    _text("Learned feature signal", 20, TEXT, bold=True),
                    _text(
                        "Separate from geometric ratings. One prediction, plus region-occlusion checks.",
                        12,
                        MUTED,
                    ),
                    _card(
                        _text("Front-facing model input", 13, TEXT, bold=True),
                        _text(
                            feature[0] if feature else "No front photo selected",
                            11,
                            GREEN if feature else MUTED,
                        ),
                        _button(
                            "Choose a feature photo",
                            icon=ft.Icons.UPLOAD_FILE_ROUNDED,
                            action=pick_action("features"),
                            disabled=busy,
                        ),
                        _text(
                            "If omitted, the uploaded front photo is reused.", 11, MUTED
                        ),
                        bgcolor=RAISED,
                    ),
                    _button(
                        "Run feature rating" if not ios_shell else "SCUT: iOS port pending",
                        icon=ft.Icons.BIOTECH_ROUNDED,
                        on_click=analyze_action("features"),
                        primary=True,
                        disabled=ios_shell or busy or feature is None,
                    ),
                ]
            )

        return body

    def workspace() -> list[ft.Control]:
        meta = {
            "photos": (
                "Step 01 / Input",
                "Start with two clear photos",
                "A guided workspace for explainable geometry and an independent AI signal.",
            ),
            "align": (
                "Step 02 / Geometry",
                "Review the landmark map",
                "Inspect detection readiness before reading a geometric report.",
            ),
            "results": (
                "Step 03 / Results",
                "Your analysis, explained",
                "Look beyond the headline: inspect every measurement and category.",
            ),
            "features": (
                "Independent / AI",
                "Explore the learned signal",
                "The model response is separate from measured facial geometry.",
            ),
        }
        eyebrow, heading, description = meta[active_tab]
        content: list[ft.Control] = [
            ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            _eyebrow(eyebrow),
                            _text(heading, 26, TEXT, bold=True),
                            _text(description, 12, MUTED),
                        ],
                        spacing=7,
                        expand=True,
                    ),
                    _tag("LOCAL · PRIVATE", color=GREEN),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            )
        ]

        if active_tab == "photos":
            content.extend(
                [
                    _card(
                        _eyebrow("Measure with context"),
                        _text(
                            "Turn two portraits into one explainable face map.",
                            25,
                            TEXT,
                            bold=True,
                        ),
                        _text(
                            "Upload each view, run the on-device detectors, and explore geometry "
                            "without sending your images to an API.",
                            13,
                            MUTED,
                        ),
                        ft.Row(
                            controls=[
                                _tag("01  Photos"),
                                _tag("02  Align"),
                                _tag("03  Results"),
                            ],
                            wrap=True,
                            spacing=8,
                        ),
                        bgcolor=RAISED,
                        padding=24,
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            ft.Container(
                                col={"xs": 12, "lg": 6},
                                content=_card(
                                    _eyebrow("Front portrait"),
                                    _photo_stage(
                                        photos.get("front"),
                                        "Front",
                                        large=False,
                                        preview=previews.get("front"),
                                    ),
                                    _text(
                                        "Straight-on · even lighting · eyes level",
                                        11,
                                        MUTED,
                                    ),
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "lg": 6},
                                content=_card(
                                    _eyebrow("Side portrait"),
                                    _photo_stage(
                                        photos.get("side"),
                                        "Side",
                                        large=False,
                                        preview=previews.get("side"),
                                    ),
                                    _text(
                                        "Full 90° profile · forehead to chin", 11, MUTED
                                    ),
                                ),
                            ),
                        ],
                        spacing=12,
                        run_spacing=12,
                    ),
                    _card(
                        _eyebrow("Input intelligence"),
                        _text("Better photos, calmer geometry.", 18, TEXT, bold=True),
                        *[
                            _text("✓ " + line, 12, MUTED)
                            for line in (
                                "Keep the camera level and avoid head tilt.",
                                "Use soft, even lighting and a clear face outline.",
                                "Include the full jaw, chin, forehead, and ear.",
                                "Use JPG, PNG, or WebP, up to 20 MB per photo.",
                            )
                        ],
                    ),
                ]
            )
        elif active_tab in ("align", "results"):
            response = responses.get(active_profile)
            right: list[ft.Control]
            if response is None:
                right = [
                    _card(
                        _eyebrow("Live analysis"),
                        _text("Waiting for analysis", 20, TEXT, bold=True),
                        _text(
                            "Choose a photo, place points in Align, and calculate offline.",
                            12,
                            MUTED,
                        ),
                        _button(
                            "Run analysis",
                            on_click=analyze_action(active_profile),
                            primary=True,
                            disabled=busy or active_profile not in photos,
                        ),
                    )
                ]
            elif active_tab == "align":
                right = [
                    _card(
                        _eyebrow("Landmark readiness"),
                        _text(
                            f"{response.get('landmark_count', 0)} detected points",
                            22,
                            TEXT,
                            bold=True,
                        ),
                        _text(
                            "Ready for results"
                            if response.get("data") is not None
                            else "Incomplete landmark map",
                            13,
                            GREEN if response.get("data") is not None else AMBER,
                        ),
                    ),
                ]
                if response.get("data") is None:
                    right.extend(
                        _report_body(response, active_profile, metric_order)[:1]
                    )
                else:
                    right.append(
                        _button(
                            "Open measurements",
                            on_click=lambda _: switch_tab("results"),
                            primary=True,
                        )
                    )
            else:
                right = _report_body(response, active_profile, metric_order)

            content.append(
                ft.ResponsiveRow(
                    controls=[
                        ft.Container(
                            col={"xs": 12, "lg": 6},
                            content=_card(
                                ft.Row(
                                    controls=[
                                        _tag(PROFILE_NAMES[active_profile]),
                                        _tag(
                                            "Loaded"
                                            if active_profile in photos
                                            else "No image",
                                            color=GREEN
                                            if active_profile in photos
                                            else AMBER,
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                ),
                                _photo_stage(
                                    photos.get(active_profile),
                                    PROFILE_NAMES[active_profile],
                                    large=not compact_screen(),
                                    preview=previews.get(active_profile),
                                ),
                                _text(
                                    "Use the Align tab to edit points directly on the portrait.",
                                    11,
                                    MUTED,
                                ),
                                bgcolor=PANEL,
                            ),
                        ),
                        ft.Container(
                            col={"xs": 12, "lg": 6},
                            content=ft.Column(controls=right, spacing=12),
                        ),
                    ],
                    spacing=14,
                    run_spacing=14,
                )
            )
        else:
            feature_image = feature_photo or photos.get("front")
            content.append(
                ft.ResponsiveRow(
                    controls=[
                        ft.Container(
                            col={"xs": 12, "lg": 6},
                            content=_card(
                                _eyebrow("Model input"),
                                _photo_stage(
                                    feature_image,
                                    "AI feature portrait",
                                    large=not compact_screen(),
                                    preview=previews.get("features")
                                    if feature_photo
                                    else previews.get("front"),
                                ),
                                _text(
                                    "Scores are model estimates, not ground truth.",
                                    11,
                                    MUTED,
                                ),
                            ),
                        ),
                        ft.Container(
                            col={"xs": 12, "lg": 6},
                            content=ft.Column(
                                controls=_feature_body(responses.get("features")),
                                spacing=12,
                            ),
                        ),
                    ],
                    spacing=14,
                    run_spacing=14,
                )
            )

        content.append(
            _text(
                "Geometry and learned features are separate project-specific signals, "
                "not objective measures of a person's value.",
                11,
                MUTED,
            )
        )
        return content

    def view_signature(key: tuple[str, str | None]) -> tuple:
        """Only invalidate UI whose underlying inputs actually changed."""
        tab, profile = key
        shared = (compact_screen(), gender.value, ethnicity.value, busy)
        front_photo = photos.get("front")
        side_photo = photos.get("side")
        selected_photo = photos.get(profile) if profile else None
        if tab == "photos":
            return shared + (
                id(front_photo),
                id(side_photo),
                id(previews.get("front")),
                id(previews.get("side")),
            )
        if tab == "features":
            return shared + (
                id(feature_photo),
                id(front_photo),
                id(responses.get("features")),
                id(previews.get("features")),
                id(previews.get("front")),
            )
        assert profile is not None
        report = responses.get(profile)
        points = manual_points[profile]
        return shared + (
            id(selected_photo),
            id(previews.get(profile)),
            id(report),
            metric_order if tab == "results" else None,
            manual_selected[profile] if tab == "align" else None,
            len(manual_history[profile]) if tab == "align" else None,
            tuple((p, v["x"], v["y"]) for p, v in points.items())
            if tab == "align"
            else None,
        )

    def render() -> None:
        nonlocal last_view
        sync_status_controls()
        key = (
            active_tab,
            active_profile if active_tab in ("align", "results") else None,
        )
        signature = view_signature(key)
        if view_signatures.get(key) != signature:
            # Build only the stale view, never all six. Reused views keep stable
            # widget identities and Flutter image/report state.
            sidebar_column = sidebar_views[key].content
            workspace_column = workspace_views[key].content
            assert isinstance(sidebar_column, ft.Column)
            assert isinstance(workspace_column, ft.Column)
            sidebar_column.controls = sidebar()
            workspace_column.controls = workspace()
            view_signatures[key] = signature
        last_view_before_switch = last_view
        if last_view != key:
            if last_view is not None:
                sidebar_views[last_view].visible = False
                workspace_views[last_view].visible = False
            sidebar_views[key].visible = True
            workspace_views[key].visible = True
            last_view = key
        index = next((idx for idx, item in enumerate(TABS) if item[0] == active_tab), 0)
        bottom_navigation.selected_index = index
        rail_navigation.selected_index = index
        previous_layout = last_compact
        reflow()
        if previous_layout != last_compact:
            # Desktop/mobile breakpoint changes the parent layout.
            page.update()
        else:
            # Flet supports targeted patches; do not walk unrelated cached views.
            patches: list[ft.Control] = [
                sidebar_views[key], workspace_views[key],
                bottom_navigation, rail_navigation,
                status_label, status_progress, status_idle,
            ]
            if last_view_before_switch is not None and last_view_before_switch != key:
                patches.extend(
                    (sidebar_views[last_view_before_switch], workspace_views[last_view_before_switch])
                )
            page.update(*patches)

    page.appbar = ft.AppBar(
        title=ft.Row(
            controls=[
                ft.Container(
                    content=_text("FQ", 14, CYAN_DARK, bold=True),
                    alignment=ft.Alignment.CENTER,
                    bgcolor=CYAN,
                    width=36,
                    height=36,
                    border_radius=12,
                ),
                _text("mini-faceIQ", 19, TEXT, bold=True),
            ],
            spacing=11,
        ),
        actions=[
            ft.IconButton(
                icon=ft.Icons.HOME_ROUNDED,
                tooltip="Open photos",
                on_click=lambda _: switch_tab("photos"),
            ),
        ],
        bgcolor=BG,
        center_title=False,
    )

    def on_resize(_event) -> None:
        # Rebuild controls only when crossing the compact-layout breakpoint.
        if compact_screen() != last_compact:
            render()
        elif not compact_screen():
            height = max(390, (page.height or 680) - 112)
            if rail_host.height != height:
                rail_host.height = height
                page.update()

    page.on_resize = on_resize

    status_card = ft.Container(
        content=ft.Column(
            controls=[
                _eyebrow("System state"),
                status_label,
                status_progress,
                status_idle,
            ],
            spacing=10,
        ),
        padding=14,
        bgcolor=SURFACE,
        border_radius=12,
    )
    # Global status is deliberately not duplicated across cached views.
    # A Flet control must have exactly one live parent.
    sidebar_content.controls.append(status_card)
    page.add(ft.SafeArea(content=root, expand=True))
    render()


if __name__ == "__main__":
    ft.run(main)
