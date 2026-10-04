from pathlib import Path

from table_ronde.tools import read_file, search_codebase


def test_search_codebase_posix_and_encoding(tmp_path: Path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    target_file = sub / "test.txt"
    # Write UTF-8 with BOM
    target_file.write_bytes(b"\xef\xbb\xbfKEYWORD in BOM file\n")

    res = search_codebase.invoke({"query": "KEYWORD", "path": str(tmp_path)})
    assert "subdir/test.txt" in res
    assert "\\" not in res
    assert "KEYWORD in BOM file" in res


def test_search_codebase_ignores_binaries(tmp_path: Path):
    dll_file = tmp_path / "lib.dll"
    dll_file.write_bytes(b"KEYWORD in binary dll\n")

    res = search_codebase.invoke({"query": "KEYWORD", "path": str(tmp_path)})
    assert "lib.dll" not in res


def test_read_file_handles_utf8_bom(tmp_path: Path):
    f = tmp_path / "bom.txt"
    f.write_bytes(b"\xef\xbb\xbfline 1\nline 2\n")

    content = read_file.invoke({"filepath": str(f)})
    assert "line 1" in content
    assert "\ufeff" not in content
