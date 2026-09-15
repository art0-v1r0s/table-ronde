import os
from pathlib import Path

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

MAX_TOTAL_CHARS = 40_000
MAX_FILE_CHARS = 5_000


def is_binary(file_path: Path) -> bool:
    try:
        with open(file_path, "tr", encoding="utf-8") as check_file:
            check_file.read(1024)
            return False
    except Exception:
        return True


def scan_project(project_path: str | Path) -> str:
    root = Path(project_path).resolve()
    if not root.exists():
        raise ValueError(f"Le chemin spécifié n'existe pas: {project_path}")

    if root.is_file():
        try:
            content = root.read_text(encoding="utf-8", errors="ignore")
            return f"--- Fichier: {root.name} ---\n{content[:MAX_FILE_CHARS]}"
        except Exception as e:
            return f"Erreur de lecture du fichier {root.name}: {e}"

    tree_lines = [f"Structure du projet : {root.name}/"]
    file_contents = []
    total_chars = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Filtrer les dossiers ignorés
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]

        rel_dir = Path(dirpath).relative_to(root)
        level = len(rel_dir.parts) if rel_dir != Path(".") else 0
        indent = "  " * level
        if rel_dir != Path("."):
            tree_lines.append(f"{indent}📁 {rel_dir.name}/")

        sub_indent = "  " * (level + 1)
        for f in sorted(filenames):
            if f.startswith("."):
                continue
            file_path = Path(dirpath) / f
            if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
                continue

            tree_lines.append(f"{sub_indent}📄 {f}")

            if total_chars < MAX_TOTAL_CHARS:
                if not is_binary(file_path):
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        if content.strip():
                            snippet = content[:MAX_FILE_CHARS]
                            rel_file_path = file_path.relative_to(root)
                            file_contents.append(f"\n--- Fichier: {rel_file_path} ---\n{snippet}")
                            total_chars += len(snippet)
                    except Exception:
                        pass

    summary = "\n".join(tree_lines) + "\n\n=== CONTENU DES FICHIERS PRINCIPAUX ===\n" + "\n".join(file_contents)
    if total_chars >= MAX_TOTAL_CHARS:
        summary += "\n\n[Attention: Le contenu a été tronqué pour respecter la limite de contexte]"

    return summary
