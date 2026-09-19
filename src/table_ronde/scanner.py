import os
from pathlib import Path

import pathspec

EXCLUDE_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".bin",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
}

EXCLUDE_EXTENSIONS: set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".lock",
}

HIGH_PRIORITY_NAMES: set[str] = {
    "readme.md",
    "pyproject.toml",
    "package.json",
    "cargo.toml",
    "go.mod",
    "makefile",
    "dockerfile",
    "main.py",
    "app.py",
    "cli.py",
    "index.ts",
    "index.js",
    "orchestrator.py",
    "agents.py",
}

HIGH_PRIORITY_EXTS: set[str] = {
    ".py",
    ".ts",
    ".js",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".sql",
}

MAX_TOTAL_CHARS = 60_000
MAX_FILE_CHARS = 6_000


def is_binary(file_path: Path) -> bool:
    try:
        with open(file_path, "tr", encoding="utf-8") as check_file:
            check_file.read(1024)
            return False
    except Exception:
        return True


def get_gitignore_spec(root: Path) -> pathspec.PathSpec | None:
    gitignore_path = root / ".gitignore"
    if gitignore_path.is_file():
        try:
            lines = gitignore_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            return pathspec.PathSpec.from_lines("gitignore", lines)
        except Exception:
            return None
    return None


def calculate_file_priority(rel_path: Path) -> int:
    name_lower = rel_path.name.lower()
    ext_lower = rel_path.suffix.lower()
    score = 0

    if name_lower in HIGH_PRIORITY_NAMES:
        score += 100
    if ext_lower in HIGH_PRIORITY_EXTS:
        score += 50
    if "test" in name_lower or "spec" in name_lower:
        score -= 30
    if len(rel_path.parts) == 1:
        score += 20
    elif len(rel_path.parts) == 2:
        score += 10

    return score


def scan_project(project_path: str | Path) -> str:
    root = Path(project_path).resolve()
    if not root.exists():
        raise ValueError(f"The specified path does not exist: {project_path}")

    if root.is_file():
        try:
            content = root.read_text(encoding="utf-8", errors="ignore")
            return f"--- File: {root.name} ---\n{content[:MAX_FILE_CHARS]}"
        except Exception as e:
            return f"Error reading file {root.name}: {e}"

    spec = get_gitignore_spec(root)

    tree_lines = [f"Project structure: {root.name}/"]
    scanned_files: list[
        tuple[int, Path, Path]
    ] = []  # (priority, rel_file_path, abs_file_path)

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)

        # Filter base excluded dirs + gitignore
        filtered_dirs = []
        for d in dirnames:
            if d in EXCLUDE_DIRS or d.startswith("."):
                continue
            rel_subdir = rel_dir / d if rel_dir != Path(".") else Path(d)
            if spec and (
                spec.match_file(str(rel_subdir)) or spec.match_file(f"{rel_subdir}/")
            ):
                continue
            filtered_dirs.append(d)
        dirnames[:] = filtered_dirs

        level = len(rel_dir.parts) if rel_dir != Path(".") else 0
        indent = "  " * level
        if rel_dir != Path("."):
            tree_lines.append(f"{indent}📁 {rel_dir.name}/")

        sub_indent = "  " * (level + 1)
        for f in sorted(filenames):
            if f.startswith("."):
                continue
            file_path = Path(dirpath) / f
            rel_file = rel_dir / f if rel_dir != Path(".") else Path(f)

            if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
                continue

            if spec and spec.match_file(str(rel_file)):
                continue

            tree_lines.append(f"{sub_indent}📄 {f}")

            if not is_binary(file_path):
                prio = calculate_file_priority(rel_file)
                scanned_files.append((prio, rel_file, file_path))

    # Sort files by decreasing priority
    scanned_files.sort(key=lambda item: item[0], reverse=True)

    file_contents = []
    total_chars = 0

    for prio, rel_file, abs_file in scanned_files:
        if total_chars >= MAX_TOTAL_CHARS:
            break
        try:
            content = abs_file.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                snippet = content[:MAX_FILE_CHARS]
                file_contents.append(
                    f"\n--- File ({prio} pts): {rel_file} ---\n{snippet}"
                )
                total_chars += len(snippet)
        except Exception:
            pass

    summary = (
        "\n".join(tree_lines)
        + "\n\n=== MAIN FILE CONTENTS (RAG Priority) ===\n"
        + "\n".join(file_contents)
    )
    if total_chars >= MAX_TOTAL_CHARS:
        summary += "\n\n[Warning: Content was intelligently selected and truncated to fit context limits]"

    return summary
