from pathlib import Path

def asPath(path):
    return path if path is Path else Path(path)

def require_dir(path):
    path = asPath(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def require_parent_dir(path):
    path = asPath(path)
    return require_dir(path)

