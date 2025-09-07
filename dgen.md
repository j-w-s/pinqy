# dgen - declarative data generation

```
.------..------..------..------.
|d.--. ||g.--. ||e.--. ||n.--. |
| :/\: || :/\: || (\/) || :(): |
| (__) || :\/: || :\/: || ()() |
| '--'d|| '--'g|| '--'e|| '--'n|
`------'`------'`------'`------'
```

a schema-based data generation library powered by faker and numpy for creating realistic, structured test data.

## core concepts

`dgen` works by interpreting a python dictionary or json object as a schema. this schema defines the structure and data types of the data you want to generate.

-   **declarative schema**: you define *what* your data looks like, not *how* to create it.
-   **faker integration**: seamlessly uses faker providers for realistic data like names, addresses, and dates.
-   **schema inference**: can automatically generate a schema by analyzing an existing json object, making it easy to get started.
-   **reproducible data**: an optional seed can be provided to generate the exact same dataset every time.

## quick start

define a schema and use it to generate a list of user objects.

```python
import dgen

# 1. define the data schema
user_schema = {
    'user_id': ('uuid4', {}),
    'username': 'user_name',
    'status': {'_qen_provider': 'choice', 'from': ['active', 'inactive', 'pending']},
    'posts': [{
        '_qen_items': {
            'title': ('bs', {}),
            'body': ('paragraph', {'nb_sentences': 5})
        },
        '_qen_count': (0, 3) # each user will have 0 to 3 posts
    }]
}

# 2. create a generator from the schema with a seed for reproducibility
generator = dgen.from_schema(user_schema, seed=123)

# 3. generate 5 user objects and convert to a list
users = generator.take(5).to.list()

# the 'users' variable now contains a list of 5 generated user dictionaries
print(users[0]['username'])
# output: 'jennifer80'
```

---

## api reference

### factory functions

| function | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `from_schema()` | `schema: any`,<br/>`seed: int = none` | creates a generator from a python dictionary schema. | `schema = {'id': 'pyint', 'name': 'word'}`<br/>`gen = dgen.from_schema(schema, seed=42)` |
| `from_json()` | `json_input: str|path`,<br/>`seed: int = none` | creates a generator by inferring a schema from an example json object, string, or file path. | `json_str = '{"id": 1, "active": true}'`<br/>`gen = dgen.from_json(json_str, seed=1)` |

### terminal operations

| method | parameters | description | example |
| :--- | :--- | :--- | :--- |
| `.take()` | `count: int` | generates a specified number of objects. returns a `pinqy.enumerable` for further processing. | `data = gen.take(100)`<br/>`data_list = data.to.list()` |

---

## schema definition guide

the schema is a dictionary that maps keys to value generators.

### faker providers

use faker provider names as strings. for providers that take arguments, use a tuple `(provider_name, kwargs_dict)`.

```python
# schema
{
    'name': 'name',                                 # simple provider
    'sentence': ('sentence', {'nb_words': 5}),      # provider with arguments
    'number': ('pyint', {'min_value': 0, 'max_value': 100})
}
```

### list generation

to generate a list of items, use a list containing a single dictionary. this dictionary defines the schema for the list items and the number of items to generate via special keys.

-   **`_qen_items`**: the schema for each individual item in the list.
-   **`_qen_count`**: the number of items to generate. can be an `int` for a fixed count or a tuple `(min, max)` for a random count within a range.

```python
# schema
{
    'tags': [{
        '_qen_items': 'word',      # each item in the list will be a random word
        '_qen_count': (3, 5)       # generate between 3 and 5 tags
    }]
}
```

### dgen providers

for custom logic, `dgen` provides its own special providers. these are defined in a dictionary with a `_qen_provider` key.

| provider | keys | description | example |
| :--- | :--- | :--- | :--- |
| **`choice`** | `from: list` | randomly selects one value from the given `from` list. | `{'status': {'_qen_provider': 'choice', 'from': ['a', 'b', 'c']}}` |
| **`ref`** | `key: str`,<br/>`format: str = none` | references a value already generated in the *same object*. useful for creating dependent fields. | `{'first': 'first_name', 'full': {'_qen_provider': 'ref', 'key': 'first'}}` |
| **`lambda`** | `func: str` | evaluates a python lambda function string. gets a `context` dict of already-generated fields. | `{'total': {'_qen_provider': 'lambda', 'func': "lambda ctx: ctx['price'] * 1.07"}}` |
| **`literal`** | `value: any` | returns a fixed, literal value. useful for constants or types like `none`. | `{'version': {'_qen_provider': 'literal', 'value': '1.0'}}` |