# fui - functional ui for pyqt6
# a declarative, component-based, pure-functional ui toolkit for pyqt6.
# it maps a state dictionary to a ui tree, inspired by the elm architecture.
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QPushButton,
                             QListWidget, QTextEdit, QFrame)
from PyQt6.QtGui import QKeySequence, QShortcut
from typing import Dict, Any, Callable, List, Tuple

# --- core types ---
component = Tuple[str, Dict[str, Any], List['component']]

# --- the application runner ---
class app:
    _instance = None # singleton for the qapplication

    def __init__(self, initial_state: Dict, page: 'page', keymap: Dict):
        if not app._instance:
            app._instance = QApplication(sys.argv)
        self.app_instance = app._instance

        self.state = initial_state
        self.page = page
        self.keymap = keymap

        self.window = QMainWindow()
        self.window.setWindowTitle("fui app")
        self.window.setGeometry(100, 100, 1024, 768)

        for key_str, action in self.keymap.items():
            shortcut = QShortcut(QKeySequence(key_str), self.window)
            shortcut.activated.connect(lambda a=action: self._handle_action(a))

    def _handle_action(self, action: str):
        if not action: return
        self.state = self.page.update(self.state, action)
        self._update_view()
        if self.state.get('should_quit', False):
            self.app_instance.quit()

    def _update_view(self):
        view_tree = self.page.view(self.state)
        new_central_widget = self._build_widget(self.window, view_tree)
        self.window.setCentralWidget(new_central_widget)

    def run(self):
        self._update_view()
        self.window.show()
        self.app_instance.exec()

    # --- internal widget builder ---
    # passes children to the builder, which is responsible for them.
    def _build_widget(self, parent: QWidget, comp: component) -> QWidget:
        comp_type, props, children = comp
        builder = getattr(self, f'_build_{comp_type}', None)
        if not builder:
            raise NameError(f"fui component type '{comp_type}' not found.")
        # the builder function is now responsible for handling the children
        widget = builder(parent, props, children)
        return widget

    # --- component builders (new robust design) ---
    def _build_vstack(self, p, props, children):
        container = QFrame(p)
        layout = QVBoxLayout()
        container.setLayout(layout)
        for child_comp in children:
            child_widget = self._build_widget(container, child_comp)
            flex = child_comp[1].get('flex', 0)
            layout.addWidget(child_widget, stretch=flex)
        return container

    def _build_hstack(self, p, props, children):
        container = QFrame(p)
        layout = QHBoxLayout()
        container.setLayout(layout)
        for child_comp in children:
            child_widget = self._build_widget(container, child_comp)
            flex = child_comp[1].get('flex', 0)
            layout.addWidget(child_widget, stretch=flex)
        return container

    def _build_boxview(self, p, props, children):
        box = QGroupBox(p)
        box.setTitle(props.get('title', ''))
        layout = QVBoxLayout() # boxview always arranges its single child vertically
        box.setLayout(layout)
        if children: # build the single child
            child_widget = self._build_widget(box, children[0])
            layout.addWidget(child_widget, stretch=1)
        return box

    # leaf components ignore the 'children' argument
    def _build_text(self, p, props, children):
        label = QLabel(props.get('content', ''), p)
        label.setWordWrap(True)
        return label

    def _build_button(self, p, props, children):
        btn = QPushButton(props.get('text', ''), p)
        btn.clicked.connect(lambda: self._handle_action(props.get('on_click')))
        return btn

    def _build_listpanel(self, p, props, children):
        widget = QListWidget(p)
        widget.addItems(props.get('items', []))
        selected = props.get('selected_index', -1)
        if selected >= 0:
            widget.setCurrentRow(selected)
        return widget

    def _build_articlepanel(self, p, props, children):
        widget = QFrame(p)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # build internal components manually
        title = QLabel(props.get('title', ''))
        subtitle = QLabel(props.get('subtitle', ''))
        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(props.get('body', ''))

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(body, stretch=1)
        return widget

# --- public api (component factory functions) ---
def page(view: Callable, update: Callable) -> 'page':
    class page_container:
        def __init__(self, view_fn, update_fn):
            self.view = view_fn
            self.update = update_fn
    return page_container(view, update)

# layout
def vstack(children: List[component], **props) -> component: return ('vstack', props, children)
def hstack(children: List[component], **props) -> component: return ('hstack', props, children)
# containers
def boxview(child: component, **props) -> component: return ('boxview', props, [child])
# primitives
def text(content: str, **props) -> component: return ('text', {'content': content, **props}, [])
def button(text: str, on_click: str, **props) -> component: return ('button', {'text': text, 'on_click': on_click, **props}, [])
# specialized
def listpanel(**props) -> component: return ('listpanel', props, [])
def articlepanel(**props) -> component: return ('articlepanel', props, [])