from pathlib import Path


WORKSPACE = Path("workspace")


def create_workspace():
    WORKSPACE.mkdir(exist_ok=True)
    return str(WORKSPACE)


def list_files():
    create_workspace()

    files = []

    for path in WORKSPACE.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(WORKSPACE)))

    return files


def read_file(file_path):
    path = WORKSPACE / file_path

    if not path.exists():
        return f"File not found: {file_path}"

    return path.read_text(encoding="utf-8")


def write_file(file_path, content):
    path = WORKSPACE / file_path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return f"File written successfully: {file_path}"