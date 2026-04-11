from pathlib import Path


ROOT = Path(".")
OUTPUT_FILE = "project_structure.txt"

# ką praleisti
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".DS_Store",
}

IGNORE_FILES = {
    ".DS_Store",
}


def should_ignore(path: Path) -> bool:
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    if path.name in IGNORE_FILES:
        return True
    return False


def build_tree(path: Path, prefix: str = "") -> list[str]:
    lines = []

    try:
        entries = sorted(
            [p for p in path.iterdir() if not should_ignore(p)],
            key=lambda p: (p.is_file(), p.name.lower())
        )
    except PermissionError:
        return [prefix + "[Permission denied]"]

    total = len(entries)

    for i, entry in enumerate(entries):
        connector = "└── " if i == total - 1 else "├── "
        lines.append(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if i == total - 1 else "│   "
            lines.extend(build_tree(entry, prefix + extension))

    return lines


def main():
    root = ROOT.resolve()
    lines = [root.name]
    lines.extend(build_tree(root))

    text = "\n".join(lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Išsaugota į {OUTPUT_FILE}")


if __name__ == "__main__":
    main()