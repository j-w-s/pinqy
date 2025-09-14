from .styles import style, borders
from . import components as c

def dark_theme():
    return {
        "app": style(fg=(220, 220, 220), bg=(30, 20, 40)),
        "focused": style(bg=(80, 70, 100)),
        "selected": style(bg=(60, 50, 80)),
        c.text: style(fg=(200, 200, 255)),
        c.button: style(padding=(0, 2, 0, 2), bg=(60, 60, 90)),
        c.input_field: style(padding=(0, 1, 0, 1), bg=(20, 10, 30), border=borders.single)
    }