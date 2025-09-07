### dgen.py README

a schema-based data generation library powered by faker and numpy for creating realistic, structured test data.

## installation

include the `dgen.py` file in your project. it has dependencies on `faker`, `numpy`, and `pinqy`.

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

## api reference

### factory functions

#### `from_schema(schema: any, seed: optional[int] = none)`
creates a data generator from a python dictionary schema.

-   **`schema`**: a python dictionary defining the structure of the data to be generated.
-   **`seed`**: an optional integer to ensure the generated data is reproducible.
-   **returns**: a `_schemaprovider` instance with a `.take()` method.

```python
object_schema = {
    'id': ('pyint', {'min_value': 1, 'max_value': 100}),
    'name': 'word'
}
# create the generator
generator = dgen.from_schema(object_schema, seed=42)
```

#### `from_json(json_input: union[str, path, textio], seed: optional[int] = none)`
creates a data generator by inferring a schema from an example json object.

-   **`json_input`**: can be a json string, a `pathlib.path` object to a `.json` file, or a file-like object. if the json contains a list of objects, the first object is used as the template.
-   **`seed`**: an optional integer for reproducibility.
-   **returns**: a `_schemaprovider` instance.

```python
# from a json string
json_string = '{"id": 1, "product_name": "widget", "active": true}'
generator = dgen.from_json(json_string, seed=1)

# from a file
# assuming 'data/example.json' exists
generator = dgen.from_json('data/example.json', seed=1)
```

### terminal operations

#### `.take(count: int)`
generates a specified number of objects based on the schema. this is called on the object returned by a factory function.

-   **`count`**: the number of objects to generate.
-   **returns**: a `pinqy.enumerable` for further processing.

```python
# generate 100 objects
data = dgen.from_schema(object_schema).take(100)

# materialize the data into a list
data_list = data.to.list()
```

## schema definition guide

the schema is a dictionary that maps keys to value generators.

### faker providers

use faker provider names as strings. for providers that take arguments, use a tuple `(provider_name, kwargs_dict)`.

```python
schema = {
    'name': 'name',                                 # simple provider
    'sentence': ('sentence', {'nb_words': 5}),      # provider with arguments
    'number': ('pyint', {'min_value': 0, 'max_value': 100})
}
```

### list generation

to generate a list of items, use a list containing a single dictionary. this dictionary defines the schema for the list items and the number of items to generate via special keys.

-   **`_qen_items`**: the schema for each individual item in the list.
-   **`_qen_count`**: the number of items to generate. can be an integer for a fixed count or a tuple `(min, max)` for a random count.

```python
schema = {
    'tags': [{
        '_qen_items': 'word',      # each item in the list will be a random word
        '_qen_count': (3, 5)       # generate between 3 and 5 tags
    }]
}
```

### dgen providers

for custom logic, `dgen` provides its own special providers via the `_qen_provider` key.

#### `choice`
randomly selects one value from a given list.

```python
schema = {
    'priority': {
        '_qen_provider': 'choice',
        'from': ['low', 'medium', 'high']
    }
}
```

#### `ref`
references a value that has already been generated in the *same object*. this is useful for creating dependent fields. an optional `format` string can be provided.

```python
schema = {
    'first_name': 'first_name',
    'full_name': {
        '_qen_provider': 'ref',
        'key': 'first_name',
        'format': '{} baker' # optional format string
    }
}
# possible result: {'first_name': 'john', 'full_name': 'john baker'}
```

#### `lambda`
evaluates a python lambda function to compute a value. the `context` dictionary, containing already-generated fields for the current object, is available.

```python
schema = {
    'price': ('pyint', {'min_value': 10, 'max_value': 100}),
    'tax': ('pyfloat', {'min_value': 0.05, 'max_value': 0.20}),
    'total_cost': {
        '_qen_provider': 'lambda',
        'func': "lambda context: context['price'] * (1 + context['tax'])"
    }
}
```

#### `literal`
returns a fixed, literal value. useful for constants or for types that are not automatically handled (like `none`).

```python
schema = {
    'version': {
        '_qen_provider': 'literal',
        'value': '1.0'
    }
}
```