import pytest
from pathlib import Path
from table_ronde.scanner import scan_project, is_binary


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
