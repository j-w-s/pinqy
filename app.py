"""
blog example application for fui
this demonstrates the same elm-like pattern, now rendering a native gui.
the application logic remains pure and separate from the view.
"""
import dgen
from pinqy import P
import fui

# --- application state ---
POST_SCHEMA = {
    'title': ('bs', {}), 'author': 'name', 'published_date': ('date_this_year', {}),
    'body': [{'_qen_items': ('paragraph', {'nb_sentences': 10}), '_qen_count': (5, 10)}],
    'comments': [{'_qen_items': {'author': 'name', 'text': 'sentence'}, '_qen_count': (0, 8)}]
}
def get_initial_state():
    posts = dgen.from_schema(POST_SCHEMA, seed=42).take(100).to.list()
    for post in posts: post['body'] = '\n\n'.join(post['body'])
    return {
        'posts': posts,
        'selected_index': 0,
        'should_quit': False,
    }

# --- keymap (pyqt key sequence to logical action mapping) ---
KEY_MAP = {
    'Up': 'nav_up',
    'Down': 'nav_down',
    'PgUp': 'pgup',
    'PgDown': 'pgdn',
    'q': 'quit',
    'Esc': 'quit',
}

# --- update logic (pure state transitions based on actions) ---
def update_app_state(state, action):
    new_state = state.copy()
    post_count = len(state['posts'])
    page_size = 15 # an arbitrary page size for pgup/pgdn

    if action == 'nav_up':
        new_state['selected_index'] = max(0, state['selected_index'] - 1)
    elif action == 'nav_down':
        new_state['selected_index'] = min(post_count - 1, state['selected_index'] + 1)
    elif action == 'pgup':
        new_state['selected_index'] = max(0, state['selected_index'] - page_size)
    elif action == 'pgdn':
        new_state['selected_index'] = min(post_count - 1, state['selected_index'] + page_size)
    elif action == 'quit':
        new_state['should_quit'] = True

    return new_state

# --- view logic (declarative ui composition) ---
def view_app(state):
    selected_post = state['posts'][state['selected_index']]

    post_titles = P(state['posts']).select_with_index(lambda p, i: f"{i+1:03d}. {p['title']}").to.list()

    comment_body = (P(selected_post['comments'])
        .select(lambda c: f"> {c['text']}\n  - {c['author']}\n")
        .to.aggregate(lambda a,b:a+b, "") or "(no comments)")

    full_content_body = selected_post['body'] + "\n\n---\n\ncomments:\n" + comment_body

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
                body=full_content_body
            )
        )
    ])

# --- main execution ---
if __name__ == "__main__":
    blog_app = fui.app(
        initial_state=get_initial_state(),
        page=fui.page(
            view=view_app,
            update=update_app_state
        ),
        keymap=KEY_MAP
    )
    blog_app.run()