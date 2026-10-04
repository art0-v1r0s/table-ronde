from pathlib import Path

from table_ronde.scanner import (
    calculate_file_priority,
    get_gitignore_spec,
    is_binary,
    scan_project,
)


def test_is_binary(tmp_path: Path):
    text_file = tmp_path / "test.txt"
    text_file.write_text("Hello, World!")
    assert not is_binary(text_file)

    bin_file = tmp_path / "test.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    assert is_binary(bin_file)


def test_get_gitignore_spec(tmp_path: Path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.log\nnode_modules/")

    spec = get_gitignore_spec(tmp_path)
    assert spec is not None
    assert spec.match_file("test.log")
    assert spec.match_file("node_modules/index.js")
    assert not spec.match_file("test.py")


def test_calculate_file_priority():
    assert calculate_file_priority(Path("readme.md")) > 100
    assert calculate_file_priority(Path("src/main.py")) >= 50
    assert calculate_file_priority(Path("test_something.py")) < 50  # penalty for tests


def test_scan_repository_with_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("ignored_dir/\n*.secret")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "secret.txt").write_text("hidden")
    (tmp_path / "app.secret").write_text("hidden")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")

    result = scan_project(tmp_path)
    assert "main.py" in result
    assert "secret.txt" not in result
    assert "app.secret" not in result

