# fui - functional ui for pyqt6

```
    _________  .__  .__
    \_   ___ \ |  | |__|
    /    \  \/ |  | |  |
    \     \____|  |_|  |
     \______  /|____/__|
            \/
```

a declarative, component-based, pure-functional ui toolkit for building python desktop applications with pyqt6.

## core concepts

fui is built on a simple but powerful architecture inspired by the elm programming language. this pattern separates your application into three distinct parts:

-   **state**: a single dictionary that holds all the data for your application.
-   **view**: a function that takes the current state and returns a tree of ui components. it's a declarative mapping of state to ui.
-   **update**: a "pure function" that takes the current state and an action (like a key press or button click) and returns a *new* state. it contains all your application's logic.

the entire application is a loop:
1.  the `view` function describes the ui for the current `state`.
2.  a user interaction (key press, mouse click) is mapped to a logical `action`.
3.  the `update` function is called with the current `state` and the `action`.
4.  the `update` function returns a brand new `state`.
5.  the loop repeats, re-rendering the view with the new state.

this one-way data flow makes applications predictable, testable, and easy to reason about. the ui is always just a function of the state (`ui = f(state)`).

## quick start: a complete example

this example demonstrates the core pattern used in `app.py`.

### 1. define the application state

start by defining the initial state of your application in a single dictionary.

```python
# app.py
def get_initial_state():
    """returns the starting state for the application."""
    return {
        'posts': [{'title': '...'}, {'title': '...'}],
        'selected_index': 0,
        'should_quit': false,
    }
```

### 2. map events to actions

create a dictionary that translates pyqt key sequences into logical actions that your `update` function will understand.

```python
# app.py
key_map = {
    'up': 'nav_up',
    'down': 'nav_down',
    'q': 'quit',
    'esc': 'quit',
}
```

### 3. write the update logic

this is the heart of your application. it's a pure function that takes the current state and an action, and returns a completely new state. no side effects or ui manipulation should happen here.

```python
# app.py
def update_app_state(state, action):
    """handles state transitions based on actions."""
    new_state = state.copy()
    post_count = len(state['posts'])

    if action == 'nav_up':
        new_state['selected_index'] = max(0, state['selected_index'] - 1)
    elif action == 'nav_down':
        new_state['selected_index'] = min(post_count - 1, state['selected_index'] + 1)
    elif action == 'quit':
        new_state['should_quit'] = true

    # notice how this function knows nothing about pyqt, tkinter, or terminals.
    # it's just pure python logic.
    return new_state
```

### 4. declare the view

this function declaratively builds your ui as a tree of components based on the current state. it doesn't create any widgets; it just describes what the ui *should* look like as a simple data structure (a tuple).

```python
# app.py
import fui
from pinqy import p

def view_app(state):
    """maps the current state to a tree of ui components."""
    selected_post = state['posts'][state['selected_index']]
    post_titles = p(state['posts']).select(lambda p: p['title']).to.list()

    return fui.hstack(flex=1, children=[
        fui.boxview(
            flex=1,
            title=f"posts ({state['selected_index']+1}/{len(state['posts'])})",
            child=fui.listpanel(
                items=post_titles,
                selected_index=state['selected_index']
            )
        ),
        fui.boxview(
            flex=2,
            title=selected_post['title'],
            child=fui.articlepanel(
                subtitle=f"by {selected_post['author']}",
                body=selected_post['body']
            )
        )
    ])
```

### 5. run the application

finally, bring all the pieces together in the application runner. the `fui.app` class manages the main loop, rendering engine, event handling, and state management for you.

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

## component reference

### application runner

| function | parameters | description |
| :--- | :--- | :--- |
| `fui.app()` | `initial_state`, `page`, `keymap` | the main application class. it orchestrates the entire process. |
| `fui.page()` | `view`, `update` | a container that logically groups a `view` function with its `update` logic. |

### layout components

these components control the position and size of their children.

| function | parameters | description |
| :--- | :--- | :--- |
| `vstack()` | `children: list`, `flex: int` | arranges child components vertically. |
| `hstack()` | `children: list`, `flex: int` | arranges child components horizontally. |

### container components

these components wrap a single child, often adding decoration like a border and title.

| function | parameters | description |
| :--- | :--- | :--- |
| `boxview()` | `child: component`, `title: str`, `flex: int` | draws a titled group box around its single child component. |

### primitive components

these components render content directly.

| function | parameters | description |
| :--- | :--- | :--- |
| `text()` | `content: str` | displays a simple, word-wrapped block of text. |
| `button()` | `text: str`, `on_click: str` | displays a button. when clicked, it emits the `on_click` action string. |

### specialized components

these are pre-built, more complex components for common ui patterns.

| function | parameters | description |
| :--- | :--- | :--- |
| `listpanel()` | `items: list[str]`, `selected_index: int` | renders a scrollable list of items, highlighting the selected one. |
| `articlepanel()` | `title: str`, `subtitle: str`, `body: str` | a component designed to show a title, subtitle, and a body of scrollable text. |