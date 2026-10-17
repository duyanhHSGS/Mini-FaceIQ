import flet as ft

from flet_face_landmarker import FletFaceLandmarker


def main(page: ft.Page):
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    page.add(

                ft.Container(height=150, width=300, alignment = ft.Alignment.CENTER, bgcolor=ft.Colors.PURPLE_200, content=FletFaceLandmarker(
                    tooltip="My new FletFaceLandmarker Control tooltip",
                    value = "My new FletFaceLandmarker Flet Control",
                ),),

    )


ft.run(main)
