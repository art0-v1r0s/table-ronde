from pathlib import Path

from table_ronde.scanner import is_binary, scan_project


def test_is_binary(tmp_path: Path):
    text_file = tmp_path / "hello.txt"
    text_file.write_text("Hello world", encoding="utf-8")
    assert not is_binary(text_file)

    bin_file = tmp_path / "data.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\xff")
    assert is_binary(bin_file)


def test_scan_project(tmp_path: Path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "main.py").write_text("print('test')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test Project", encoding="utf-8")

    result = scan_project(tmp_path)
    assert "Structure du projet" in result
    assert "main.py" in result
    assert "README.md" in result
    assert "print('test')" in result


def test_scan_project_with_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("ignored_folder/\n*.secret\n", encoding="utf-8")

    ignored_dir = tmp_path / "ignored_folder"
    ignored_dir.mkdir()
    (ignored_dir / "secret_file.py").write_text("SECRET_KEY = '123'", encoding="utf-8")

    secret_file = tmp_path / "pass.secret"
    secret_file.write_text("password123", encoding="utf-8")

    valid_file = tmp_path / "pyproject.toml"
    valid_file.write_text("[project]\nname='test'", encoding="utf-8")

    result = scan_project(tmp_path)
    assert "ignored_folder" not in result
    assert "pass.secret" not in result
    assert "pyproject.toml" in result

