def get_kwargs(name, kwargs):
    if name not in kwargs:
        raise ValueError(f"Expected {name} in kwargs.")
    return kwargs[name]
