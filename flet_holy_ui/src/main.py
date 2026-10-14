"""Mini-FaceIQ Flet desktop UI — direct Python inference, no HTTP or Flask server."""

from __future__ import annotations

import asyncio
import json

import flet as ft
from local_engine import MAX_IMAGE_BYTES, AnalysisError, analyze_photo

BACKGROUND = "#101827"
CARD = "#1B2638"
MUTED = "#A8B5C9"
ACCENT = "#57B7FF"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _result_controls(result: dict) -> list[ft.Control]:
    data = result.get("data")
    count = result.get("landmark_count")
    output: list[ft.Control] = []

    if count is not None:
        output.append(ft.Text(f"Detected landmarks: {count}", color=MUTED))

    if data is None:
        output.append(ft.Text(result.get("warning", "Analysis could not be completed."), color="#FFCA80"))
        missing = result.get("missing_ids", [])
        if missing:
            output.append(ft.Text("Missing IDs: " + ", ".join(map(str, missing)), selectable=True, size=12))
        output.append(
            ft.Text(
                "Automatic detection can omit required points. The original browser "
                "UI supports manually adjusting missing landmarks.",
                color=MUTED,
            )
        )
        return output

    if not isinstance(data, dict):
        return [ft.Text(str(data), selectable=True)]

    if result.get("mode") != "features":
        for key, label in (
            ("overallScore", "Overall"),
            ("frontScore", "Front"),
            ("sideScore", "Side"),
            ("harmonyScore", "Harmony"),
        ):
            score = data.get(key)
            if not isinstance(score, (int, float)):
                continue
            output.append(
                ft.Column(
                    spacing=5,
                    controls=[
                        ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Text(label, weight=ft.FontWeight.BOLD),
                                ft.Text(f"{score:.2f} / 10"),
                            ],
                        ),
                        ft.ProgressBar(value=max(0.0, min(1.0, score / 10))),
                    ],
                )
            )

        for key, label in (
            ("topStrengths", "Higher-scoring metrics"),
            ("topWeaknesses", "Lower-scoring metrics"),
        ):
            values = data.get(key)
            if isinstance(values, list) and values:
                output.append(ft.Text(label, weight=ft.FontWeight.BOLD, size=16))
                output.append(ft.Text(", ".join(str(x) for x in values)))

        measurements = data.get(
            "frontMeasurements" if result.get("mode") == "front" else "sideMeasurements",
            [],
        )
        if isinstance(measurements, list) and measurements:
            output.append(ft.Text("Measurements", size=18, weight=ft.FontWeight.BOLD))
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    continue
                name = str(measurement.get("name", "Measurement"))
                value = _format_value(measurement.get("value", "—"))
                unit = str(measurement.get("unit", ""))
                score = measurement.get("score")
                suffix = f"  •  {_format_value(score)}/10" if isinstance(score, (int, float)) else ""
                output.append(
                    ft.Container(
                        padding=10,
                        border_radius=8,
                        bgcolor=BACKGROUND,
                        content=ft.Text(f"{name}: {value} {unit}{suffix}", selectable=True),
                    )
                )
    else:
        output.append(ft.Text("Feature-rating response", size=18, weight=ft.FontWeight.BOLD))
        output.append(
            ft.Text(
                json.dumps(data, ensure_ascii=False, indent=2)[:12000],
                selectable=True,
                font_family="monospace",
                size=12,
            )
        )

    return output or [ft.Text("The analyzer returned an empty result.")]


def main(page: ft.Page) -> None:
    page.title = "Mini-FaceIQ Holy UI"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BACKGROUND
    page.padding = 12

    image_name: str | None = None
    image_bytes: bytes | None = None

    mode = ft.Dropdown(
        label="Analysis",
        value="front",
        options=[
            ft.DropdownOption(key="front", text="Front profile"),
            ft.DropdownOption(key="side", text="Side profile"),
            ft.DropdownOption(key="features", text="Feature rating"),
        ],
        expand=True,
    )
    gender = ft.Dropdown(
        label="Reference set",
        value="male",
        options=[
            ft.DropdownOption(key="male", text="Male"),
            ft.DropdownOption(key="female", text="Female"),
        ],
        expand=True,
    )
    ethnicity = ft.Dropdown(
        label="Norm group",
        value="asian",
        options=[
            ft.DropdownOption(key="asian", text="Asian"),
            ft.DropdownOption(key="caucasian", text="Caucasian"),
            ft.DropdownOption(key="south_asian", text="South Asian"),
        ],
        expand=True,
    )

    status = ft.Text(
        "Select a photo and tap Analyze. Processing runs locally in Python.",
        color=MUTED,
        selectable=True,
    )
    selected_label = ft.Text("No photo selected", color=MUTED)
    preview = ft.Image(src="", width=330, height=260, fit=ft.BoxFit.CONTAIN, visible=False)
    results = ft.Column(spacing=12)
    spinner = ft.ProgressRing(visible=False, width=24, height=24)

    async def on_analyze(_event) -> None:
        if image_bytes is None or image_name is None:
            status.value = "Pick a face photo first."
            status.color = "#FFCA80"
            page.update()
            return

        analyze_button.disabled = True
        spinner.visible = True
        status.value = "Running local face detection and measurements..."
        status.color = MUTED
        results.controls.clear()
        page.update()

        try:
            analysis = await asyncio.to_thread(
                analyze_photo,
                mode.value or "front",
                image_name,
                image_bytes,
                gender.value or "male",
                ethnicity.value or "asian",
            )
            results.controls.extend(_result_controls(analysis))
            completed = analysis.get("data") is not None
            status.value = "Local analysis finished." if completed else "Partial result: missing landmarks."
            status.color = "#86E4B6" if completed else "#FFCA80"
        except AnalysisError as exc:
            status.value = str(exc)
            status.color = "#FFCA80"
        except Exception as exc:  # noqa: BLE001 — surface local model errors, keep UI responsive.
            status.value = f"Local inference failed ({type(exc).__name__}): {str(exc)[:400]}"
            status.color = "#FFCA80"
        finally:
            analyze_button.disabled = False
            spinner.visible = False
            page.update()

    def on_file_result(event: ft.FilePickerResultEvent) -> None:
        nonlocal image_name, image_bytes
        if not event.files:
            return
        selected = event.files[0]
        if not selected.bytes:
            status.value = "Could not read this photo. Please select it again."
            status.color = "#FFCA80"
            page.update()
            return
        if len(selected.bytes) > MAX_IMAGE_BYTES:
            status.value = "Photo is larger than 20 MB; choose a smaller image."
            status.color = "#FFCA80"
            page.update()
            return

        image_name = selected.name
        image_bytes = selected.bytes
        preview.src = image_bytes
        preview.visible = True
        selected_label.value = f"{image_name}  •  {len(image_bytes) / 1024:.0f} KB"
        status.value = "Photo loaded. Ready for local analysis."
        status.color = MUTED
        results.controls.clear()
        page.update()

    picker = ft.FilePicker(on_result=on_file_result)
    page.services.append(picker)

    analyze_button = ft.Button(
        content="Analyze locally",
        icon=ft.Icons.AUTO_AWESOME,
        on_click=on_analyze,
    )

    layout = ft.Column(
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        controls=[
            ft.Text("Mini-FaceIQ", size=28, weight=ft.FontWeight.BOLD, color=ACCENT),
            ft.Text("Holy Flet UI  •  Local Python inference  •  No API server", size=13, color=MUTED),
            ft.Container(
                padding=16,
                bgcolor=CARD,
                border_radius=16,
                content=ft.Column(
                    spacing=12,
                    controls=[
                        ft.Text("1 · Choose your photo", size=18, weight=ft.FontWeight.BOLD),
                        mode,
                        ft.Row(controls=[gender, ethnicity], spacing=8),
                        ft.Button(
                            content="Choose face photo",
                            icon=ft.Icons.PHOTO_LIBRARY,
                            action=ft.PickFiles(
                                picker,
                                allow_multiple=False,
                                file_type=ft.FilePickerFileType.CUSTOM,
                                allowed_extensions=["jpg", "jpeg", "png", "webp"],
                                with_data=True,
                            ),
                        ),
                        selected_label,
                        ft.Row(controls=[preview], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Row(controls=[analyze_button, spinner], spacing=12),
                        status,
                    ],
                ),
            ),
            ft.Container(
                padding=16,
                bgcolor=CARD,
                border_radius=16,
                content=ft.Column(
                    spacing=12,
                    controls=[
                        ft.Text("2 · Local results", size=18, weight=ft.FontWeight.BOLD),
                        results,
                    ],
                ),
            ),
            ft.Text(
                "Images are processed locally without API uploads. Installed local "
                "ML models perform detection and scoring. Scores reflect project-specific "
                "rules, not objective human worth.",
                size=11,
                color=MUTED,
            ),
        ],
    )
    page.add(ft.SafeArea(content=layout, expand=True))


if __name__ == "__main__":
    ft.run(main)
