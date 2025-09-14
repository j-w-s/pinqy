from dataclasses import dataclass, field, replace
from typing import Tuple, Literal, Optional

color = Tuple[int, int, int]
padding = Tuple[int, int, int, int]  # top, right, bottom, left

@dataclass(frozen=True)
class border_style:
    top: str = '─'
    bottom: str = '─'
    left: str = '│'
    right: str = '│'
    top_left: str = '┌'
    top_right: str = '┐'
    bottom_left: str = '└'
    bottom_right: str = '┘'

class borders:
    none = border_style('', '', '', '', '', '', '', '')
    single = border_style()
    double = border_style('═', '═', '║', '║', '╔', '╗', '╚', '╝')
    heavy = border_style('━', '━', '┃', '┃', '┏', '┓', '┗', '┛')
    rounded = border_style('─', '─', '│', '│', '╭', '╮', '╰', '╯')

@dataclass(frozen=True)
class style:
    fg: Optional[color] = None
    bg: Optional[color] = None
    border: Optional[border_style] = None
    padding: Optional[padding] = None

    def merge(self, other: 'style') -> 'style':
        return style(
            fg=other.fg or self.fg,
            bg=other.bg or self.bg,
            border=other.border or self.border,
            padding=other.padding or self.padding
        )