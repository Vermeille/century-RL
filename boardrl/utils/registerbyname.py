import inspect
from collections.abc import Mapping
from typing import Any, get_type_hints


def parse_spec(spec: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse ``name,key=value`` specs without doing registry-specific work."""
    if isinstance(spec, Mapping):
        kwargs = dict(spec)
        try:
            name = kwargs.pop("name")
        except KeyError as exc:
            raise ValueError("Spec mapping requires a 'name' field") from exc
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Spec name must be a non-empty string")
        return name.strip(), kwargs

    if not isinstance(spec, str):
        raise ValueError(f"Invalid spec: {spec!r} (type: {type(spec).__name__})")

    name, *parts = spec.split(",")
    name = name.strip()
    if not name:
        raise ValueError("Spec name cannot be empty")

    kwargs: dict[str, str] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid spec argument: {part!r}")
        if key in kwargs:
            raise ValueError(f"Duplicate spec argument: {key}")
        kwargs[key] = value.strip()
    return name, kwargs


def _identity(value):
    return value


def _coerce(value, arg_type):
    if arg_type is bool:
        if isinstance(value, bool):
            return value
        if value not in ("True", "False"):
            raise ValueError("expected True or False")
        return value == "True"
    return arg_type(value)


class RegisterByName:
    def __init__(self):
        self.registry = {}
        self.required = {}

    def copy(self):
        new_register = RegisterByName()
        new_register.registry = self.registry.copy()
        new_register.required = {name: args.copy() for name, args in self.required.items()}
        return new_register

    def register(self, name, args_from=None):
        def foo(cls):
            source = cls if args_from is None else args_from
            if inspect.isclass(source) and "__init__" in source.__dict__:
                params = inspect.signature(source.__init__).parameters
                annotated = source.__init__
            else:
                params = inspect.signature(source).parameters
                annotated = source

            try:
                type_hints = get_type_hints(annotated)
            except (NameError, TypeError):
                type_hints = {}

            relevant = {
                arg_name: param
                for arg_name, param in params.items()
                if arg_name != "self"
                and param.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
            arg_info = {
                arg_name: (
                    type_hints.get(
                        arg_name,
                        param.annotation
                        if param.annotation != inspect.Parameter.empty
                        else _identity,
                    ),
                    param.default
                    if param.default != inspect.Parameter.empty
                    else None,
                )
                for arg_name, param in relevant.items()
            }

            self.registry[name] = (cls, arg_info)
            self.required[name] = {
                arg_name
                for arg_name, param in relevant.items()
                if param.default == inspect.Parameter.empty
            }
            cls._registry_name = name
            return cls

        return foo

    def update(self, other: "RegisterByName"):
        self.registry.update(other.registry)
        self.required.update(
            {name: args.copy() for name, args in other.required.items()}
        )
        return self

    def __contains__(self, spec):
        name, _ = parse_spec(spec)
        return name in self.registry

    def __call__(self, spec, **provided_args):
        name, args = parse_spec(spec)

        if name not in self.registry:
            raise ValueError(f"Unknown class: {name}")

        klass, arg_info = self.registry[name]
        unknown = set(args) - set(arg_info)
        if unknown:
            unknown_args = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown argument(s) for {name}: {unknown_args}")

        init_args = {}
        for arg_name, (arg_type, default) in arg_info.items():
            if arg_name in provided_args:
                init_args[arg_name] = provided_args[arg_name]
                continue
            if arg_name in args:
                value = args[arg_name]
                try:
                    init_args[arg_name] = _coerce(value, arg_type)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid value for {arg_name} in {name}: {value!r}"
                    ) from exc
                continue
            if arg_name in self.required[name]:
                raise ValueError(f"Missing required argument {arg_name} for {name}")
            init_args[arg_name] = default

        return klass(**init_args)

    def display(self):
        for name, (_, args) in self.registry.items():
            rendered = name
            for arg, (_, default) in args.items():
                rendered += f",{arg}={'?' if arg in self.required[name] else default}"
            print(rendered)
