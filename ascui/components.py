from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Any, Callable, Optional
from .styles import style

# base component class
@dataclass(frozen=True)
class component:
    key: Optional[str] = None
    flex: int = 0
    _style: Optional[style] = None

    def with_style(self, **kwargs) -> component:
        new_style = style(**kwargs)
        merged_style = self._style.merge(new_style) if self._style else new_style
        return replace(self, _style=merged_style)

    def with_key(self, key: str) -> component:
        return replace(self, key=key)

    def grow(self, flex_val: int = 1) -> component:
        return replace(self, flex=flex_val)

# component types
@dataclass(frozen=True)
class v_container(component):
    children: List[component] = field(default_factory=list)

@dataclass(frozen=True)
class h_container(component):
    children: List[component] = field(default_factory=list)

@dataclass(frozen=True)
class text(component):
    content: str = ""

@dataclass(frozen=True)
class button(text):
    action: Optional[tuple] = None

@dataclass(frozen=True)
class input_field(text):
    value: str = ""
    placeholder: str = ""
    on_change: Optional[Callable[[str], tuple]] = None

@dataclass(frozen=True)
class animation(component):
    frames_fn: Callable[[float], List[str]] = field(default_factory=lambda: lambda t: [])

# component factory functions
def v(*children: component) -> v_container:
    return v_container(children=list(children))

def h(*children: component) -> h_container:
    return h_container(children=list(children))

def txt(content: Any = "") -> text:
    return text(content=str(content))

def anim(frames_fn: Callable[[float], List[str]]) -> animation:
    return animation(frames_fn=frames_fn)

def spacer() -> text:
    return text(content="").grow()

# composite/helper components
def btn(label: str, action_type: str, payload: Any = None) -> component:
    return button(content=f" {label} ", key=label, action=(action_type, payload or label))

def lst(items: List[Any], selected: int, renderer: Callable[[Any], str]) -> component:
    def render_item(i, item):
        line = renderer(item)
        comp = txt(f" {line} ")
        if i == selected: return comp.with_key("selected_item")
        return comp
    return v(*[render_item(i, item) for i, item in enumerate(items)])

def inp(value: str, placeholder: str, on_change: Callable[[str], tuple]) -> component:
    display_text = value or placeholder
    fg_color = (255, 255, 255) if value else (100, 100, 100)
    return input_field(
        content=f" {display_text} ", value=value, placeholder=placeholder, on_change=on_change
    ).with_style(fg=fg_color)