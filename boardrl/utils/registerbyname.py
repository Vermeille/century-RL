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

    def register(self, name):
        def foo(cls):
            # Extract the argument names, types, and defaults from the __init__ method
            if "__init__" in cls.__dict__:
                sig = inspect.signature(cls.__init__)
                params = sig.parameters
                arg_info = {
                    name: (
                        param.annotation
                        if param.annotation != inspect.Parameter.empty
                        else lambda x: x,
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

    def __call__(self, descr_string, **provided_args):
        name, *arg_list = descr_string.split(",")
        args = {arg.split("=")[0]: arg.split("=")[1] for arg in arg_list}

        if name not in self.registry:
            raise ValueError(f"Unknown class: {descr_string}")

        klass, arg_info = self.registry[name]
        init_args = {}

        for arg_name in args.keys():
            assert arg_name in arg_info, f"Unknown argument {arg_name} for {name}"

        for arg_name, (arg_type, default) in arg_info.items():
            if arg_name in self.arg_readers:
                init_args[arg_name] = self.arg_readers[arg_name](
                    args.get(arg_name, None), default, provided_args.get(arg_name, None)
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

        return klass(**init_args)

    def display(self):
        for fun, args in self.registry.items():
            fun_display = fun
            for arg, (arg_type, default) in args[1].items():
                if default == inspect.Parameter.empty:
                    default = "?"
                fun_display += f",{arg}={default}"
            print(fun_display)
