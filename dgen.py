'''
.------..------..------..------.
|d.--. ||g.--. ||e.--. ||n.--. |
| :/\: || :/\: || (\/) || :(): |
| (__) || :\/: || :\/: || ()() |
| '--'d|| '--'g|| '--'e|| '--'n|
`------'`------'`------'`------'
'''

import json
import re
import numpy as np
from faker import Faker
from pinqy import from_iterable, Enumerable
from typing import Any, Dict, Optional, Union, TextIO
from pathlib import Path


class Generator:
    """schema interpreter."""

    def __init__(self, seed: Optional[int] = None):
        self._fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

    def _resolve_faker_method(self, method_name: str, kwargs: Dict = {}) -> Any:
        try:
            method = getattr(self._fake, method_name)
            return method(**kwargs)
        except AttributeError:
            raise ValueError(f"faker has no provider '{method_name}'")

    def _resolve_provider(self, config: Dict, context: Dict) -> Any:
        provider = config["_qen_provider"]
        if provider == "ref":
            key = config["key"]
            if key not in context:
                raise ValueError(f"reference to '{key}' not found in current context.")
            value = context[key]
            if "format" in config:
                return config["format"].format(value)
            return value

        elif provider == "choice":
            # convert numpy's choice result to a native python type
            choice_result = self._rng.choice(config["from"])
            return choice_result.item() if hasattr(choice_result, 'item') else choice_result

        elif provider == "lambda":
            func_str = config["func"]
            lambda_func = eval(func_str)
            return lambda_func(context)

        elif provider == "literal":
            if "value" not in config:
                raise ValueError("_qen_provider 'literal' requires a 'value' key.")
            return config["value"]

        else:
            raise ValueError(f"unknown _qen_provider: '{provider}'")

    def create(self, schema: Any, context: Optional[Dict] = None) -> Any:
        # if context is none, it becomes an empty dict.
        current_context = context or {}

        # main recursive logic
        if isinstance(schema, dict):
            if "_qen_provider" in schema:
                return self._resolve_provider(schema, current_context)

            # create a new object, building a local context sequentially
            generated_obj = {}
            for k, v in schema.items():
                # merge parent context with the current local context
                # this allows refs to look up (into parent) and sideways (into local)
                merged_context = {**current_context, **generated_obj}
                generated_obj[k] = self.create(v, merged_context)
            return generated_obj

        if isinstance(schema, list):
            if not schema: return []
            item_schema = schema[0]
            # pass the parent's context down into list items
            # the schema for a list item is expected to be a dict that contains the count
            # if it's not a dict, it won't have a count, and the default is used
            count = self._get_count(item_schema)
            # the actual schema for the item is inside the '_qen_items' key if it exists, otherwise it's the schema itself
            actual_item_schema = item_schema.get('_qen_items', item_schema)
            return [self.create(actual_item_schema, current_context) for _ in range(count)]

        if isinstance(schema, str):
            if hasattr(self._fake, schema):
                return self._resolve_faker_method(schema)
            return schema  # otherwise, it's a literal string.

        if isinstance(schema, tuple) and len(schema) == 2 and isinstance(schema[1], dict):
            return self._resolve_faker_method(schema[0], schema[1])

        return schema

    def _get_count(self, item_schema: Any) -> int:
        count = 5 # default count
        if isinstance(item_schema, dict) and "_qen_count" in item_schema:
            count_config = item_schema["_qen_count"]
            if isinstance(count_config, int):
                count = count_config
            elif isinstance(count_config, (list, tuple)) and len(count_config) == 2:
                low, high = count_config
                count = self._rng.integers(low, high, endpoint=True)
        return count


class _SchemaProvider:
    def __init__(self, schema: Any, seed: Optional[int] = None):
        self._schema = schema
        self._generator = Generator(seed)

    def take(self, count: int) -> Enumerable:
        def data_func():
            # the initial call to create() will have context=none
            return [self._generator.create(self._schema) for _ in range(count)]

        return from_iterable(data_func())


def from_schema(schema: Any, seed: Optional[int] = None) -> _SchemaProvider:
    return _SchemaProvider(schema, seed)

# --- schema inference ---

def _infer_schema(obj: Any, key_hint: Optional[str] = None) -> Any:
    """recursively infers a dgen schema from a python object."""
    if isinstance(obj, dict):
        # recurse into dictionaries
        return {k: _infer_schema(v, k) for k, v in obj.items()}

    if isinstance(obj, list):
        # if list is not empty, infer schema from the first item
        if not obj:
            return []

        # the schema for a list is a list containing one dictionary.
        # this dictionary holds the schema for the items and the count.
        item_schema = _infer_schema(obj[0])
        list_schema_wrapper = {
            "_qen_items": item_schema,
            "_qen_count": len(obj)
        }
        return [list_schema_wrapper]

    if isinstance(obj, str):
        # apply heuristics to guess the type of string data
        lower_key = key_hint.lower() if key_hint else ""

        # specific patterns first
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', obj, re.IGNORECASE):
            return 'uuid4'
        if '@' in obj and '.' in obj and ' ' not in obj:
            return 'email'
        if obj.startswith(('http://', 'https://')):
            return 'url'

        # key-based hints
        if 'name' in lower_key:
            return 'name'
        if 'city' in lower_key:
            return 'city'
        if 'country' in lower_key:
            return 'country'
        if 'address' in lower_key:
            return 'address'

        # content-based hints
        if ' ' not in obj:
            return 'word'
        if '\n' in obj or len(obj.split()) > 10:
            return 'paragraph'
        return 'sentence'

    if isinstance(obj, bool):
        # a choice between true and false
        return {'_qen_provider': 'choice', 'from': [True, False]}

    if isinstance(obj, int):
        # an integer within a plausible range based on the example
        return 'pyint', {'min_value': obj - abs(obj), 'max_value': obj + abs(obj) or 100}

    if isinstance(obj, float):
        # a float within a plausible range
        return 'pyfloat', {'min_value': obj - abs(obj), 'max_value': obj + abs(obj) or 100.0}

    # for anything else (like none), treat it as a literal value
    return {'_qen_provider': 'literal', 'value': obj}


def from_json(json_input: Union[str, Path, TextIO], seed: Optional[int] = None) -> _SchemaProvider:
    """
    creates a data generator by inferring a schema from an example json object or file.

    :param json_input: a json string, a file path, or a file-like object.
    :param seed: an optional seed for reproducible data generation.
    :return: a schema provider with a .take() method for generating data.
    """
    if isinstance(json_input, Path):
        with json_input.open('r') as f:
            data = json.load(f)
    elif isinstance(json_input, str):
        try:
            # try to treat as a file path first
            path = Path(json_input)
            if path.is_file() and path.suffix.lower() == '.json':
                with path.open('r') as f:
                    data = json.load(f)
            else:
                # otherwise, treat as a json string
                data = json.loads(json_input)
        except (OSError, json.JSONDecodeError):
            # handle case where string is not a valid path or json
            data = json.loads(json_input)
    else:
        # assume it's a file-like object
        data = json.load(json_input)

    # if the json is a list of objects, use the first as the template
    template = data[0] if isinstance(data, list) and data else data

    # infer the schema from the template
    inferred_schema = _infer_schema(template)

    return _SchemaProvider(inferred_schema, seed)