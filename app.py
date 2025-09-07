"""
blog example application for fui v10 (the engine).
this version uses the new App/Page architecture for a clean separation of concerns.
navigation and scrolling are now fully dynamic and layout-aware.
"""
import math, dgen, fui
from pinqy import P

# --- 1. application state ---
POST_SCHEMA = {
    'title': ('bs', {}), 'author': 'name', 'published_date': ('date_this_year', {}),
    'body': [{'_qen_items': ('paragraph', {'nb_sentences': 10}), '_qen_count': (5, 10)}],
    'comments': [{'_qen_items': {'author': 'name', 'text': 'sentence'}, '_qen_count': (0, 8)}]
}
def get_initial_state():
    posts = dgen.from_schema(POST_SCHEMA, seed=42).take(100).to.list()
    for post in posts: post['body'] = '\n\n'.join(post['body'])
    return {
        'posts': posts, 'selected_index': 0, 'focused_pane': 'list',
        'list_scroll_offset': 0, 'content_scroll_offset': 0, 'should_quit': False,
    }

# --- 2. keymap (input to logical action mapping) ---
KEY_MAP = {
    repr('\x1b[A'): 'nav_up', repr(b'\xe0H'): 'nav_up',       # up arrows
    repr('\x1b[B'): 'nav_down', repr(b'\xe0P'): 'nav_down',   # down arrows
    repr('\x1b[D'): 'nav_left', repr(b'\xe0K'): 'nav_left',   # left arrows
    repr('\x1b[C'): 'nav_right', repr(b'\xe0M'): 'nav_right', # right arrows
    repr('\x1b[5~'): 'pgup', repr(b'\xe0I'): 'pgup',          # page up
    repr('\x1b[6~'): 'pgdn', repr(b'\xe0Q'): 'pgdn',          # page down
    repr('\t'): 'focus_next', repr('\r'): 'focus_content', 'q': 'quit',
    repr('\x08'): 'backspace', repr('\x7f'): 'backspace', repr(b'\b'): 'backspace',
}

# --- 3. update logic (pure state transitions based on actions) ---
def update_app_state(state, action, term_size):
    new_state = state.copy()
    pane_height = max(1, term_size.height - 4) # visible height in panels

    if state['focused_pane'] == 'list':
        post_count = len(state['posts'])
        if action == 'nav_up': new_state['selected_index'] = max(0, state['selected_index'] - 1)
        elif action == 'nav_down': new_state['selected_index'] = min(post_count - 1, state['selected_index'] + 1)
        elif action == 'pgup': new_state['selected_index'] = max(0, state['selected_index'] - pane_height)
        elif action == 'pgdn': new_state['selected_index'] = min(post_count - 1, state['selected_index'] + pane_height)
        elif action in ['focus_next', 'focus_content', 'nav_right']: new_state['focused_pane'] = 'content'
        elif action == 'backspace': new_state['should_quit'] = True

        # auto-scroll list to keep selection in view
        if new_state['selected_index'] < state.get('list_scroll_offset', 0):
            new_state['list_scroll_offset'] = new_state['selected_index']
        if new_state['selected_index'] >= state.get('list_scroll_offset', 0) + pane_height:
            new_state['list_scroll_offset'] = new_state['selected_index'] - pane_height + 1

        if state['selected_index'] != new_state['selected_index']: new_state['content_scroll_offset'] = 0

    elif state['focused_pane'] == 'content':
        post = state['posts'][state['selected_index']]
        content_width = (term_size.width * 2 // 3) - 4
        full_content = post['body'] + P(post['comments']).select(lambda c: f"\n> {c['text']}...").to.aggregate(lambda a,b:a+b,"")
        max_scroll = max(0, len(fui._wrap_text(full_content, content_width)) - pane_height)

        if action == 'nav_up': new_state['content_scroll_offset'] = max(0, state['content_scroll_offset'] - 1)
        elif action == 'nav_down': new_state['content_scroll_offset'] = min(max_scroll, state['content_scroll_offset'] + 1)
        elif action == 'pgup': new_state['content_scroll_offset'] = max(0, state['content_scroll_offset'] - pane_height)
        elif action == 'pgdn': new_state['content_scroll_offset'] = min(max_scroll, state['content_scroll_offset'] + pane_height)
        elif action in ['focus_next', 'nav_left', 'backspace']: new_state['focused_pane'] = 'list'

    if action == 'quit': new_state['should_quit'] = True
    return new_state

# --- 4. view logic (declarative ui composition) ---
def view_app(state):
    selected_post = state['posts'][state['selected_index']]

    post_titles = P(state['posts']).select_with_index(lambda p, i: f"{i+1}. {p['title']}").to.list()

    full_content_body = selected_post['body'] + "\n\n---\n\nComments:\n" + (P(selected_post['comments'])
        .select(lambda c: f"\n> {c['text']}\n  - {c['author']}\n").to.aggregate(lambda a,b:a+b, "") or "\n(No comments)")

    footer_text = ("arrows/pg: navigate | tab/enter: content | q: quit" if state['focused_pane'] == 'list'
                   else "arrows/pg: scroll | tab/bksp: list | q: quit")

    return fui.VStack(children=[
        fui.Header("fui blog reader"),
        fui.HStack(flex=1, children=[
            fui.BoxView(flex=1, props={'title': f"posts ({state['selected_index']+1}/{len(state['posts'])})", 'fg': fui.colors.cyan, 'fg_focused': fui.colors.yellow, 'is_focused': state['focused_pane'] == 'list'},
                children=[fui.ListPanel(props={'items': post_titles, 'selected_index': state['selected_index'], 'scroll_offset': state['list_scroll_offset']})]),
            fui.BoxView(flex=2, props={'title': selected_post['title'], 'fg': fui.colors.cyan, 'fg_focused': fui.colors.yellow, 'is_focused': state['focused_pane'] == 'content'},
                children=[fui.ArticlePanel(props={'subtitle': f"by {selected_post['author']}", 'body': full_content_body, 'scroll_offset': state['content_scroll_offset']})])
        ]),
        fui.Footer(footer_text)
    ])

# --- main execution ---
if __name__ == "__main__":
    blog_app = fui.App(
        initial_state=get_initial_state(),
        page=fui.Page(
            view=view_app,
            update=update_app_state
        ),
        keymap=KEY_MAP
    )
    blog_app.run(debug_keys=False) # set debug_keys=True to see raw key codes