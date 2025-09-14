import sys
import time
from PyQt6 import QtWidgets, QtGui, QtCore
from .renderer import renderer
from . import components as c


class AsciiWidget(QtWidgets.QWidget):
    def __init__(self, app_host, parent=None):
        super().__init__(parent)
        self.host = app_host
        self.renderer = renderer(self.host.font_path, self.host.font_size)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(False)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        comp_tree = self.host.page.view_fn(self.host.state)
        self.renderer.render(painter, self.host.theme, comp_tree, self.host.state.get('focus'), time.time())
        painter.end()


class app:
    def __init__(self, title, width, height, font_path="ascui/fonts/oryx-simplex.ttf", font_size=16):
        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.title, self.width, self.height = title, width, height
        self.font_path, self.font_size = font_path, font_size
        self.state, self.theme, self.page = {}, {}, None

    def with_state(self, initial_state):
        self.state = initial_state
        return self

    def with_theme(self, theme):
        self.theme = theme
        return self

    def add_page(self, name, page_instance):
        self.page = page_instance
        return self

    def _handle_key(self, event: QtGui.QKeyEvent):
        action = None
        key_text = event.text()
        qt_key = event.key()
        modifiers = event.modifiers()

        comp_tree = self.page.view_fn(self.state)
        focused_comp = next((c for c in self._get_all_comps(comp_tree) if c.key == self.state.get('focus')), None)

        if key_text and key_text in self.page.keymap:
            action = self.page.keymap[key_text]
        elif qt_key in self.page.keymap:
            action = self.page.keymap[qt_key]

        if not action and isinstance(focused_comp, c.input_field):
            current_value = focused_comp.value
            if qt_key == QtCore.Qt.Key.Key_Backspace:
                action = focused_comp.on_change(current_value[:-1])
            elif key_text.isprintable():
                action = focused_comp.on_change(current_value + key_text)

        if not action and qt_key == QtCore.Qt.Key.Key_Tab:
            action = ('focus_prev', None) if modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier else ('focus_next',
                                                                                                         None)

        if not action and qt_key == QtCore.Qt.Key.Key_Enter:
            if isinstance(focused_comp, c.button) and focused_comp.action:
                action = focused_comp.action

        if action:
            if not isinstance(action, tuple): action = (action, None)
            self.state = self.page.update_fn(self.state, action)
            self.central_widget.update()

    def _get_all_comps(self, comp):
        comps = [comp]
        if hasattr(comp, 'children'):
            for child in comp.children:
                comps.extend(self._get_all_comps(child))
        return comps

    def run(self):
        window = QtWidgets.QMainWindow()
        window.setWindowTitle(self.title)
        window.setGeometry(100, 100, self.width, self.height)
        self.central_widget = AsciiWidget(self)
        window.setCentralWidget(self.central_widget)
        window.keyPressEvent = self._handle_key
        timer = QtCore.QTimer(window)
        timer.timeout.connect(self.central_widget.update)
        timer.start(1000 // 30)
        window.show()
        sys.exit(self.qt_app.exec())


class page:
    def __init__(self, view_fn, update_fn, keymap=None):
        self.view_fn = view_fn
        self.update_fn = update_fn
        self.keymap = keymap or {}