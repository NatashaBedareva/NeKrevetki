import flet as ft

def main(page: ft.Page):
    page.title = "Видеонаблюдение"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.Colors.GREY_900
    page.padding = 20
    
    # Контейнер для изображения/видеопотока
    video_container = ft.Container(
        content=ft.Image(
            src="https://picsum.photos/800/450",
            fit="contain",
        ),
        width=800,
        height=450,
        bgcolor=ft.Colors.BLACK,
        border_radius=15,
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.Colors.BLACK38,
        ),
    )
    
    # Текст для кнопок
    calibrate_text = ft.Text("Калибровка", size=16, weight=ft.FontWeight.W_500)
    start_text = ft.Text("Старт", size=16, weight=ft.FontWeight.W_500)
    stop_text = ft.Text("Стоп", size=16, weight=ft.FontWeight.W_500)
    
    # Кнопка "Калибровка"
    calibrate_btn = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.TUNE), calibrate_text],
            spacing=10,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        width=200,
        height=45,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            bgcolor=ft.Colors.BLUE_700,
            color=ft.Colors.WHITE,
        ),
    )
    
    # Кнопка "Старт/Стоп"
    start_stop_content = ft.Row(
        [ft.Icon(ft.Icons.PLAY_ARROW), start_text],
        spacing=10,
        alignment=ft.MainAxisAlignment.CENTER,
    )
    
    start_stop_btn = ft.Button(
        content=start_stop_content,
        width=200,
        height=45,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            bgcolor=ft.Colors.GREEN_700,
            color=ft.Colors.WHITE,
        ),
    )
    
    # Функция-заглушка для смены текста кнопки
    def toggle_start_stop(e):
        if start_text.value == "Старт":
            start_text.value = "Стоп"
            start_stop_content.controls[0].name = ft.Icons.STOP
            start_stop_btn.style.bgcolor = ft.Colors.RED_700
        else:
            start_text.value = "Старт"
            start_stop_content.controls[0].name = ft.Icons.PLAY_ARROW
            start_stop_btn.style.bgcolor = ft.Colors.GREEN_700
        page.update()
    
    start_stop_btn.on_click = toggle_start_stop
    
    # Панель с кнопками
    button_row = ft.Row(
        controls=[calibrate_btn, start_stop_btn],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20,
    )
    
    # Основной макет
    page.add(
        ft.Column(
            controls=[
                ft.Container(height=20),
                ft.Row(
                    controls=[video_container],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Container(height=30),
                button_row,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )
    
    page.update()

if __name__ == "__main__":
    ft.app(target=main)