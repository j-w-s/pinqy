'''
.------..------..------..------.
|D.--. ||G.--. ||E.--. ||N.--. |
| :/\: || :/\: || (\/) || :(): |
| (__) || :\/: || :\/: || ()() |
| '--'D|| '--'G|| '--'E|| '--'N|
`------'`------'`------'`------'
'''

import numpy as np
from faker import Faker
from pinqy import from_iterable, Enumerable
from typing import Any, Dict, Optional


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
            return self._rng.choice(config["from"])

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
            return [self.create(item_schema, current_context) for _ in range(self._get_count(item_schema))]

        if isinstance(schema, str):
            if hasattr(self._fake, schema):
                return self._resolve_faker_method(schema)
            return schema  # otherwise, it's a literal string.

        if isinstance(schema, tuple) and len(schema) == 2 and isinstance(schema[1], dict):
            return self._resolve_faker_method(schema[0], schema[1])

        return schema

    def _get_count(self, item_schema: Any) -> int:
        count = 5
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
            # the initial call to create() will have context=None
            return [self._generator.create(self._schema) for _ in range(count)]

        return from_iterable(data_func())


def from_schema(schema: Any, seed: Optional[int] = None) -> _SchemaProvider:
    return _SchemaProvider(schema, seed)