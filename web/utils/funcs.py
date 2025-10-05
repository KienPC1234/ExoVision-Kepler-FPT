import importlib

def lazy_import(module_name, function_name):
    """Delays the import of a function until it is called."""
    func = None
    
    def wrapper(*args, **kwargs):
        nonlocal func
        if func:
            return func(*args, **kwargs)
        func = getattr(importlib.import_module(module_name), function_name)
        return func(*args, **kwargs)

    return wrapper