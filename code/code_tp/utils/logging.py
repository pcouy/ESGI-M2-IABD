import functools

def add_args_as_info(func):
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        self.infos = {
            "args": args,
            "kwargs": kwargs
        }
        func(*args, **kwargs)
    return _wrapper


