import os

P0 = os.environ.get("P0") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def p(*parts):
    return os.path.join(P0, *parts)


def runs_dir(name=None):
    d = p("runs") if name is None else p("runs", name)
    os.makedirs(d, exist_ok=True)
    return d


def gates_dir():
    d = p("gates")
    os.makedirs(d, exist_ok=True)
    return d


def models_dir(name=None):
    return p("models") if name is None else p("models", name)


def data_dir(*parts):
    return p("data", *parts)


def labels_dir(*parts):
    return p("labels", *parts)
