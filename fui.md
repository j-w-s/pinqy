---

# fui - functional ui for terminals

```
    _________  .__  .__
    \_   ___ \ |  | |__|
    /    \  \/ |  | |  |
    \     \____|  |_|  |
     \______  /|____/__|
            \/
```

## core concepts

fui is built on a simple but powerful architecture inspired by the elm programming language. this pattern separates your application into three distinct parts:

-   **state**: a single dictionary that holds all the data for your application.
-   **view**: a function that takes the current state and returns a tree of ui components. it's a declarative mapping of state to ui.
-   **update**: a "pure function" that takes the current state and an action (like a key press) and returns a *new* state. it contains all your application's logic.

the entire application is a loop:
1.  the `view` function renders the current `state`.
2.  the user presses a key, which is mapped to a logical `action`.
3.  the `update` function is called with the current `state` and the `action`.
4.  the `update` function returns a brand new `state`.
5.  the loop repeats with the new state.

this one-way data flow makes applications predictable, testable, and easy to reason about.

## quick start: a complete example

this example demonstrates the core pattern used in `app.py`.

### 1. define the application state

start by defining the initial state of your application in a single dictionary.

```python
# app.py
def get_initial_state():
    """returns the starting state for the application."""
    return {
        'posts': ['post one', 'post two', 'post three'],
        'selected_index': 0,
        'should_quit': false,
    }
```

### 2. map keys to actions

create a dictionary that translates raw keyboard input into logical actions that your `update` function will understand.

```python
# app.py
key_map = {
    repr('\x1b[a'): 'nav_up',    # up arrow
    repr('\x1b[b'): 'nav_down',  # down arrow
    'q': 'quit',
}
```

### 3. write the update logic

this is the heart of your application. it's a pure function that takes the current state and an action, and returns a completely new state. no side effects should happen here.

```python
# app.py
def update_app_state(state, action, term_size):
    """handles state transitions based on actions."""
    new_state = state.copy()
    post_count = len(state['posts'])

    if action == 'nav_up':
        new_state['selected_index'] = max(0, state['selected_index'] - 1)
    elif action == 'nav_down':
        new_state['selected_index'] = min(post_count - 1, state['selected_index'] + 1)
    elif action == 'quit':
        new_state['should_quit'] = true

    return new_state
```

### 4. declare the view

this function declaratively builds your ui as a tree of components based on the current state. it doesn't draw anything; it just describes what the ui *should* look like.

```python
# app.py
import fui

def view_app(state):
    """maps the current state to a tree of ui components."""
    return fui.vstack(children=[
        fui.header("my app"),
        fui.boxview(
            flex=1,
            props={
                'title': f"posts ({state['selected_index']+1}/{len(state['posts'])})",
                'is_focused': true,
                'fg_focused': fui.colors.yellow
            },
            children=[
                fui.listpanel(props={
                    'items': state['posts'],
                    'selected_index': state['selected_index']
                })
            ]
        ),
        fui.footer("up/down: navigate | q: quit")
    ])
```

### 5. run the application

finally, bring all the pieces together in the application runner. the `fui.app` class manages the main loop, input, rendering, and state management for you.

```python
# app.py
if __name__ == "__main__":
    my_app = fui.app(
        initial_state=get_initial_state(),
        page=fui.page(
            view=view_app,
            update=update_app_state
        ),
        keymap=key_map
    )
    my_app.run()
```

## application runner components

these are the top-level classes used to structure and run your app.

#### `fui.app(initial_state, page, keymap)`
the main application class. it orchestrates the entire process.

#### `fui.page(view, update)`
a container that logically groups a `view` function with its corresponding `update` logic. this allows for building multi-page applications in the future.

#### `.run(debug_keys=false)`
starts the application's main loop. set `debug_keys=true` to print the `repr()` of keys as you press them, which is useful for populating your `keymap`.

## layout components

these components control the position and size of their children.

#### `vstack(children, flex=0, padding=0)`
arranges child components vertically.
-   **`children`**: a list of child components to stack.
-   **`flex`**: if greater than 0, this component will expand to fill available vertical space relative to its siblings.

#### `hstack(children, flex=0, padding=0)`
arranges child components horizontally.
-   **`children`**: a list of child components to arrange side-by-side.
-   **`flex`**: if greater than 0, this component will expand to fill available horizontal space relative to its siblings.

## container components

these components typically have one or more children and add decorations or padding around them.

#### `boxview(children, flex=0, props={})`
draws a box with borders around its single child component.
-   **`children`**: must be a list containing exactly one child component.
-   **`props`**: a dictionary of properties:
    -   `title: str`: text to display in the top border.
    -   `is_focused: bool`: if `true`, uses the `fg_focused` color.
    -   `fg: int`: the border color (from `fui.colors`).
    -   `fg_focused: int`: the border color to use when `is_focused` is `true`.

## primitive components

these components render content directly.

#### `text(flex=0, padding=0, props={})`
displays text, with automatic wrapping.
-   **`props`**:
    -   `content: str`: the text to display.
    -   `fg: int`: foreground (text) color.
    -   `bg: int`: background color.
    -   `scroll_offset: int`: the line number to start rendering from.

#### `header(content: str)`
a semantic wrapper for a `text` component, styled as a header. provides top padding.

#### `footer(content: str)`
a semantic wrapper for a `text` component, styled as a footer. provides bottom padding.

## specialized components

these are pre-built components for common ui patterns.

#### `listpanel(props={})`
renders a scrollable list of items, highlighting the selected one.
-   **`props`**:
    -   `items: list[str]`: the list of strings to display.
    -   `selected_index: int`: the index of the item to highlight.
    -   `scroll_offset: int`: the index of the item to display at the top of the panel. your `update` logic is responsible for calculating this to keep the selection in view.

#### `articlepanel(props={})`
a component designed to show a title, subtitle, and a body of scrollable text. it's a `vstack` of `text` components internally.
-   **`props`**:
    -   `title: str`: the main title.
    -   `subtitle: str`: smaller text below the title.
    -   `body: str`: the main, multi-line content.
    -   `scroll_offset: int`: the vertical scroll position for the `body` text.

## colors

use the `fui.colors` object to style your components.

-   `fui.colors.black`
-   `fui.colors.red`
-   `fui.colors.green`
-   `fui.colors.yellow`
-   `fui.colors.blue`
-   `fui.colors.magenta`
-   `fui.colors.cyan`
-   `fui.colors.white`
-   `fui.colors.grey`
-   `fui.colors.dark_grey`
-   `fui.colors.light_grey`