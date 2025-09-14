import time
from dataclasses import dataclass, field, replace
from PyQt6 import QtGui, QtCore
from . import components as c
from .styles import style, borders


def _wrap_text(text_data: str, width: int) -> list[str]:
    if width <= 0: return text_data.split('\n')
    lines = []
    for p in text_data.split('\n'):
        line, words = [], p.split()
        for word in words:
            if len(" ".join(line + [word])) > width:
                lines.append(" ".join(line));
                line = [word]
            else:
                line.append(word)
        lines.append(" ".join(line))
    return lines


@dataclass
class layout_node:
    comp: c.component
    style: style
    x: int = 0;
    y: int = 0;
    w: int = 0;
    h: int = 0
    children: list['layout_node'] = field(default_factory=list)
    render_lines: list[str] = field(default_factory=list)


class renderer:
    def __init__(self, font_path: str, font_size: int):
        font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
        font_family = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0] if font_id >= 0 else "monospace"
        self.font = QtGui.QFont(font_family, font_size)
        metrics = QtGui.QFontMetrics(self.font)
        self.char_w, self.char_h = metrics.horizontalAdvance(' '), metrics.height()
        self.cursor_on = True

    def render(self, painter: QtGui.QPainter, theme: dict, comp_tree: c.component, focus_key: str, t: float):
        self.cursor_on = (int(t * 2) % 2 == 0)
        painter.setFont(self.font)
        layout_tree = self._build_layout_tree(theme, comp_tree, focus_key, t)
        self._measure_pass(layout_tree)
        self._arrange_pass(layout_tree, 0, 0, painter.window().width() // self.char_w,
                           painter.window().height() // self.char_h)
        self._draw_pass(painter, layout_tree, theme, focus_key)

    def _get_style_for(self, theme: dict, comp: c.component, focus_key: str) -> style:
        base_style = theme.get(type(comp), style())
        if type(comp) is c.text and comp.key == "selected_item":
            base_style = base_style.merge(theme.get("selected", style()))
        if comp.key == focus_key:
            base_style = base_style.merge(theme.get("focused", style()))
        return base_style.merge(comp._style or style())

    def _build_layout_tree(self, theme, comp, focus_key, t):
        node_style = self._get_style_for(theme, comp, focus_key)
        node = layout_node(comp=comp, style=node_style)
        if isinstance(comp, c.animation):
            node.render_lines = comp.frames_fn(t)
        if hasattr(comp, 'children'):
            node.children = [self._build_layout_tree(theme, child, focus_key, t) for child in comp.children]
        return node

    def _measure_pass(self, node: layout_node):
        for child in node.children: self._measure_pass(child)
        s = node.style
        pt, pr, pb, pl = s.padding or (0, 0, 0, 0)
        border = 2 if s.border and s.border is not borders.none else 0

        if isinstance(node.comp, c.text):
            node.w = max(len(l) for l in node.comp.content.split('\n')) if node.comp.content else 0
            node.h = len(node.comp.content.split('\n'))
        elif isinstance(node.comp, c.animation):
            node.w = max(len(l) for l in node.render_lines) if node.render_lines else 0
            node.h = len(node.render_lines)
        elif isinstance(node.comp, c.v_container):
            node.w = max((child.w for child in node.children), default=0)
            node.h = sum(child.h for child in node.children)
        elif isinstance(node.comp, c.h_container):
            node.w = sum(child.w for child in node.children)
            node.h = max((child.h for child in node.children), default=0)

        node.w += pl + pr + border
        node.h += pt + pb + border

    def _arrange_pass(self, node: layout_node, x, y, w, h):
        node.x, node.y, node.w, node.h = x, y, w, h
        s = node.style
        pt, pr, pb, pl = s.padding or (0, 0, 0, 0)
        border = 1 if s.border and s.border is not borders.none else 0
        content_x, content_y = x + pl + border, y + pt + border
        content_w = w - (pl + pr + border * 2);
        content_h = h - (pt + pb + border * 2)

        if isinstance(node.comp, c.text):
            node.render_lines = _wrap_text(node.comp.content, content_w)
        if not isinstance(node.comp, c.animation) and node.comp.flex == 0:
            node.h = len(node.render_lines) + pt + pb + border * 2

        if not node.children: return

        flex_total = sum(child.comp.flex for child in node.children)
        if isinstance(node.comp, c.v_container):
            fixed_h = sum(child.h for child in node.children if child.comp.flex == 0)
            remaining_h = content_h - fixed_h
            offset = 0
            for child in node.children:
                ch = child.h if child.comp.flex == 0 else int(
                    remaining_h * (child.comp.flex / flex_total)) if flex_total > 0 else 0
                self._arrange_pass(child, content_x, content_y + offset, content_w, ch)
                offset += child.h
        elif isinstance(node.comp, c.h_container):
            fixed_w = sum(child.w for child in node.children if child.comp.flex == 0)
            remaining_w = content_w - fixed_w
            offset = 0
            for child in node.children:
                cw = child.w if child.comp.flex == 0 else int(
                    remaining_w * (child.comp.flex / flex_total)) if flex_total > 0 else 0
                self._arrange_pass(child, content_x + offset, content_y, cw, content_h)
                offset += child.w

    def _draw_pass(self, painter, node: layout_node, theme: dict, focus_key: str):
        s, r = node.style, QtCore.QRect(node.x * self.char_w, node.y * self.char_h, node.w * self.char_w,
                                        node.h * self.char_h)
        painter.save()
        painter.setClipRect(r)

        bg_color = s.bg or theme.get("app", style()).bg
        if bg_color: painter.fillRect(r, QtGui.QColor(*bg_color))

        pen_color = s.fg or theme.get("app", style()).fg
        painter.setPen(QtGui.QColor(*pen_color))

        if s.border and s.border is not borders.none: self._draw_border(painter, node)

        pt, _, _, pl = s.padding or (0, 0, 0, 0)
        border = 1 if s.border and s.border is not borders.none else 0
        content_x, content_y = node.x + pl + border, node.y + pt + border

        if isinstance(node.comp, (c.text, c.animation)):
            for i, line in enumerate(node.render_lines):
                self._draw_text(painter, line, content_x, content_y + i)
            if isinstance(node.comp, c.input_field) and node.comp.key == focus_key and self.cursor_on:
                cursor_x = content_x + 1 + len(node.comp.value)
                self._draw_text(painter, "█", cursor_x, content_y)

        for child in node.children: self._draw_pass(painter, child, theme, focus_key)
        painter.restore()

    def _draw_text(self, p, s, cx, cy):
        p.drawText(int(cx * self.char_w), int((cy + 1) * self.char_h - (self.char_h * 0.2)), s)

    def _draw_border(self, p, n):
        b, w, h, x, y = n.style.border, n.w - 1, n.h - 1, n.x, n.y
        if not b or w < 1 or h < 1: return
        self._draw_text(p, b.top_left, x, y);
        self._draw_text(p, b.top_right, x + w, y)
        self._draw_text(p, b.bottom_left, x, y + h);
        self._draw_text(p, b.bottom_right, x + w, y + h)
        for i in range(1, w): self._draw_text(p, b.top, x + i, y); self._draw_text(p, b.bottom, x + i, y + h)
        for i in range(1, h): self._draw_text(p, b.left, x, y + i); self._draw_text(p, b.right, x + w, y + i)