"""
fui - functional ui toolkit for the terminal (v10 - The Engine)
a declarative, component-based, pure-functional ui toolkit designed for building
robust, resizable, and scalable terminal applications.
"""
import sys, os, shutil, time
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
try: import msvcrt
except ImportError: import tty, termios
from pinqy import from_iterable as P

# --- core types ---
@dataclass(frozen=True)
class Cell: char: str=' '; fg: Optional[int]=None; bg: Optional[int]=None
Frame = List[List[Cell]]
@dataclass(frozen=True)
class Box: x: int; y: int; width: int; height: int
@dataclass(frozen=True)
class Size: width: int=0; height: int=0

# --- ansi escape code helpers ---
class _ansi:
    csi='\033['; reset=f'{csi}0m'; hide_cursor=f'{csi}?25l'; show_cursor=f'{csi}?25h'
    @staticmethod
    def move_to(x, y): return f'{_ansi.csi}{y+1};{x+1}H'
    @staticmethod
    def fg(c): return f'{_ansi.csi}38;5;{c}m'
    @staticmethod
    def bg(c): return f'{_ansi.csi}48;5;{c}m'
class colors:
    black,red,green,yellow,blue,magenta,cyan,white = (16,196,46,226,21,201,51,231)
    grey,dark_grey,light_grey = (244,238,250)

# --- rendering and input (stable) ---
def _render_diff(previous, current):
    height, width = (len(current), len(current[0])) if current else (0,0)
    if height == 0: return
    cmds, fg, bg, x, y = [], None, None, -1, -1
    def move(nx, ny):
        nonlocal x, y
        if ny != y or nx != x+1: cmds.append(_ansi.move_to(nx, ny))
    for r, row in enumerate(current):
        for c, new in enumerate(row):
            old = previous[r][c] if r<len(previous) and c<len(previous[r]) else None
            if new != old:
                move(c, r); x, y = c, r
                if new.fg!=fg or new.bg!=bg: cmds.append(_ansi.reset)
                if new.fg is not None: cmds.append(_ansi.fg(new.fg))
                if new.bg is not None: cmds.append(_ansi.bg(new.bg))
                fg,bg=new.fg,new.bg; cmds.append(new.char)
    if cmds: sys.stdout.write(''.join(cmds)); sys.stdout.flush()

def _get_key() -> str:
    if os.name == 'nt':
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'): ch += msvcrt.getch()
        return repr(ch) # use repr for a consistent, unambiguous key format
    else:
        fd = sys.stdin.fileno(); old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno()); key = sys.stdin.read(1)
            if key == '\x1b':
                termios.tcsetattr(fd, termios.TCSANOW, old_settings)
                time.sleep(0.01)
                import select
                if select.select([sys.stdin], [], [], 0)[0]: key += sys.stdin.read(5)
                tty.setraw(sys.stdin.fileno())
            return repr(key) # use repr for a consistent, unambiguous key format
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# --- component and layout system ---
@dataclass
class Component:
    children: List['Component']=field(default_factory=list); padding: int=0; flex: int=0; props: Dict[str, Any]=field(default_factory=dict)
    def render(self, frame: Frame, box: Box) -> Frame:
        p_box = Box(box.x+self.padding, box.y+self.padding, box.width-2*self.padding, box.height-2*self.padding)
        return P(self.children).to.aggregate(lambda f, c: c.render(f, p_box), seed=frame)
    def get_preferred_size(self, max_size: Size) -> Size: return Size()

def _distribute_space(children, max_size, get_size_func):
    sizes = [get_size_func(c, max_size) for c in children]
    fixed = sum(s for s, c in zip(sizes, children) if c.flex == 0)
    total_flex = sum(c.flex for c in children if c.flex > 0)
    if total_flex == 0: return sizes
    rem = max(0, max_size - fixed)
    return [s if c.flex==0 else int(rem * (c.flex / total_flex)) for s, c in zip(sizes, children)]

@dataclass
class VStack(Component):
    def render(self, frame, box):
        p_box = Box(box.x+self.padding, box.y+self.padding, box.width-2*self.padding, box.height-2*self.padding)
        heights = _distribute_space(self.children, p_box.height, lambda c, m: c.get_preferred_size(Size(p_box.width, m)).height)
        y, f = p_box.y, frame
        for child, h in zip(self.children, heights):
            f = child.render(f, Box(p_box.x, y, p_box.width, h)); y += h
        return f

@dataclass
class HStack(Component):
    def render(self, frame, box):
        p_box = Box(box.x+self.padding, box.y+self.padding, box.width-2*self.padding, box.height-2*self.padding)
        widths = _distribute_space(self.children, p_box.width, lambda c, m: c.get_preferred_size(Size(m, p_box.height)).width)
        x, f = p_box.x, frame
        for child, w in zip(self.children, widths):
            f = child.render(f, Box(x, p_box.y, w, p_box.height)); x += w
        return f

# --- semantic and primitive components ---
def _paint_text(frame, x, y, text, fg, bg):
    if y < 0 or y >= len(frame): return frame
    new_row = list(frame[y])
    for i, char in enumerate(text):
        if 0 <= x+i < len(new_row): new_row[x+i] = Cell(char, fg, bg)
    new_frame = list(frame); new_frame[y] = new_row
    return new_frame

def _wrap_text(text, width):
    if width <= 0: return text.split('\n')
    lines = []
    for p in text.split('\n'):
        line, words = [], p.split()
        if not words: lines.append(''); continue
        for w in words:
            if len(' '.join(line+[w])) <= width: line.append(w)
            else: lines.append(' '.join(line)); line=[w]
        lines.append(' '.join(line))
    return lines

@dataclass
class Text(Component):
    def render(self, frame, box):
        p_box = Box(box.x+self.padding, box.y+self.padding, box.width-2*self.padding, box.height-2*self.padding)
        text=self.props.get('content',''); lines=_wrap_text(text, p_box.width)
        scroll=self.props.get('scroll_offset',0); lines=lines[scroll:]
        f = frame
        for i, line in enumerate(lines):
            if i >= p_box.height: break
            f = _paint_text(f, p_box.x, p_box.y+i, line, self.props.get('fg'), self.props.get('bg'))
        return f

@dataclass
class BoxView(Component):
    def render(self, frame, box):
        if box.width<2 or box.height<2: return frame
        is_focused=self.props.get('is_focused',False); fg=self.props.get('fg_focused') if is_focused else self.props.get('fg')
        title=self.props.get('title',''); top='┌'+('─'*(box.width-2))+'┐'; btm='└'+('─'*(box.width-2))+'┘'
        f=_paint_text(frame,box.x,box.y,top,fg,None); f=_paint_text(f,box.x,box.y+box.height-1,btm,fg,None)
        for r in range(1,box.height-1): f=_paint_text(f,box.x,box.y+r,'│',fg,None); f=_paint_text(f,box.x+box.width-1,box.y+r,'│',fg,None)
        if title: f=_paint_text(f,box.x+2,box.y,f" {title} ",fg,None)
        child_box=Box(box.x+1,box.y+1,box.width-2,box.height-2)
        return self.children[0].render(f,child_box) if self.children else f

def Header(content: str) -> Component: return Text(padding=1, props={'content': content, 'fg': colors.yellow})
def Footer(content: str) -> Component: return Text(padding=1, props={'content': content, 'fg': colors.dark_grey})

@dataclass
class ListPanel(Component):
    def render(self, frame, box):
        items = self.props.get('items', [])
        selected_index = self.props.get('selected_index', 0)
        scroll_offset = self.props.get('scroll_offset', 0)

        # dynamic page size calculation
        visible_item_count = max(0, box.height)

        visible_items = items[scroll_offset : scroll_offset + visible_item_count]

        f = frame
        for i, item_text in enumerate(visible_items):
            is_selected = (scroll_offset + i) == selected_index
            bg = colors.blue if is_selected else None
            fg = colors.white if is_selected else colors.grey
            line_text = item_text.ljust(box.width) # pad to fill background
            f = _paint_text(f, box.x, box.y + i, line_text, fg, bg)
        return f

@dataclass
class ArticlePanel(Component):
    def render(self, frame, box):
        return VStack(padding=1, children=[
            Text(props={'content': self.props.get('title', ''), 'fg': colors.white}),
            Text(props={'content': self.props.get('subtitle', ''), 'fg': colors.grey}),
            Text(flex=1, props={'content': self.props.get('body', ''), 'scroll_offset': self.props.get('scroll_offset', 0)})
        ]).render(frame, box)

# --- application runner ---
@dataclass
class Page:
    view: Callable[[Dict[str, Any]], Component]
    update: Callable[[Dict[str, Any], str, Size], Dict[str, Any]]

@dataclass
class App:
    initial_state: Dict[str, Any]
    page: Page
    keymap: Dict[str, str]

    def run(self, debug_keys=False):
        sys.stdout.write(_ansi.hide_cursor)
        state = self.initial_state
        cols, rows = shutil.get_terminal_size()
        previous = [[Cell(char='`')] * cols for _ in range(rows)]
        try:
            while not state.get('should_quit', False):
                term_size = shutil.get_terminal_size()
                state['term_size'] = Size(term_size.columns, term_size.lines) # inject size into state

                root = self.page.view(state)
                current = root.render([[Cell()]*term_size.columns for _ in range(term_size.lines)], Box(0,0,term_size.columns,term_size.lines))
                _render_diff(previous, current)
                previous = current

                key = _get_key()
                action = self.keymap.get(key)

                if debug_keys and not action: print(f"unknown key: {key}")
                if action: state = self.page.update(state, action, state['term_size'])
        finally:
            sys.stdout.write(_ansi.reset + _ansi.show_cursor)
            print("\napp exited cleanly.")