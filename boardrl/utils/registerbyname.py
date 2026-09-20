from typing import Tuple, get_type_hints
import inspect


class RegisterByName:
    def __init__(self, arg_readers=None):
        self.registry = {}
        self.arg_readers = arg_readers or {}

    def copy(self):
        new_register = RegisterByName()
        new_register.registry = self.registry.copy()
        new_register.arg_readers = self.arg_readers.copy()
        return new_register

    def register(self, name, args_from=None):
        def foo(cls):
            source = cls if args_from is None else args_from
            # If registering a class, prefer its __init__ signature
            if inspect.isclass(source) and "__init__" in source.__dict__:
                sig = inspect.signature(source.__init__)
                params = sig.parameters
                annotated = source.__init__
            else:
                sig = inspect.signature(source)
                params = sig.parameters
                annotated = source

            try:
                type_hints = get_type_hints(annotated)
            except (NameError, TypeError):
                type_hints = {}

            arg_info = None
            if params is not None:
                arg_info = {
                    name: (
                        type_hints.get(
                            name,
                            param.annotation
                            if param.annotation != inspect.Parameter.empty
                            else lambda x: x,
                        ),
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None,
                    )
                    for name, param in params.items()
                    if name != "self"
                }
            else:
                arg_info = {}

            self.registry[name] = (cls, arg_info)
            cls._registry_name = name
            return cls

        return foo

    def update(self, other: "RegisterByName"):
        self.registry.update(other.registry)
        self.arg_readers.update(other.arg_readers)
        return self

    def __contains__(self, descr):
        name, _ = self._read(descr)
        return name in self.registry

    def _read(self, descr: str | dict) -> Tuple[str, dict]:
        if isinstance(descr, str):
            name, *arg_list = descr.split(",")
            args = {arg.split("=")[0]: arg.split("=")[1] for arg in arg_list}
        elif isinstance(descr, dict):
            descr = dict(descr)
            args = descr
            name = args.pop("name")
        else:
            raise ValueError(f"Invalid description: {descr} (type: {type(descr)})")
        return name, args

    def __call__(self, descr_string, **provided_args):
        name, args = self._read(descr_string)

        if name not in self.registry:
            raise ValueError(f"Unknown class: {name}")

        klass, arg_info = self.registry[name]
        init_args = {}

        for arg_name in args.keys():
            assert arg_name in arg_info, f"Unknown argument {arg_name} for {name}"

        for arg_name, (arg_type, default) in arg_info.items():
            try:
                if arg_name in self.arg_readers:
                    init_args[arg_name] = self.arg_readers[arg_name](
                        args.get(arg_name, None),
                        default,
                        provided_args.get(arg_name, None),
                    )
                elif arg_name in provided_args:
                    init_args[arg_name] = provided_args[arg_name]
                elif arg_name in args:
                    if arg_type is bool:
                        assert args[arg_name] in ["True", "False"]
                        init_args[arg_name] = args[arg_name] == "True"
                    else:
                        init_args[arg_name] = arg_type(args[arg_name])
                else:
                    init_args[arg_name] = default
            except Exception as e:
                raise ValueError(
                    f"Error processing argument {arg_name} ({repr(arg_type)}({args[arg_name]})) for {name} : {e}"
                ) from e

        return klass(**init_args)

    def display(self):
        for fun, args in self.registry.items():
            fun_display = fun
            for arg, (_, default) in args[1].items():
                if default == inspect.Parameter.empty:
                    default = "?"
                fun_display += f",{arg}={default}"
            print(fun_display)
