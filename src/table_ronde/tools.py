import os
from pathlib import Path

from langchain_core.tools import tool

@tool
def search_codebase(query: str, path: str) -> str:
    """
    Search the codebase for a specific query using string matching.
    Useful for finding where a specific variable, class, or function is used.
    
    Args:
        query: The string to search for.
        path: The root directory to search in.
    
    Returns:
        A string containing the matched files and the relevant lines.
    """
    root_path = Path(path).resolve()
    if not root_path.exists() or not root_path.is_dir():
        return f"Error: The path '{path}' does not exist or is not a directory."
        
    results = []
    # Try to ignore common binary/build folders
    ignore_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}
    ignore_exts = {".pyc", ".so", ".exe", ".png", ".jpg", ".pdf"}
    
    match_count = 0
    max_matches = 50 # To prevent overwhelming the context
    
    try:
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
            
            for file in files:
                if Path(file).suffix.lower() in ignore_exts:
                    continue
                    
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if query in line:
                                rel_path = file_path.relative_to(root_path)
                                results.append(f"{rel_path}:{line_num}: {line.strip()}")
                                match_count += 1
                                if match_count >= max_matches:
                                    results.append(f"... (truncated after {max_matches} matches)")
                                    break
                except UnicodeDecodeError:
                    pass # Skip binary files that don't match our ignore_exts
                if match_count >= max_matches:
                    break
            if match_count >= max_matches:
                break
    except Exception as e:
        return f"Error while searching: {e!s}"
        
    if not results:
        return f"No matches found for '{query}'."
        
    return "\n".join(results)

@tool
def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """
    Read the contents of a specific file. 
    Can optionally read only a specific range of lines.
    
    Args:
        filepath: The absolute or relative path to the file.
        start_line: The line number to start reading from (1-indexed). Optional.
        end_line: The line number to end reading at (1-indexed). Optional.
    
    Returns:
        The content of the file.
    """
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return f"Error: The file '{filepath}' does not exist."
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        start = max(1, start_line) if start_line else 1
        end = min(len(lines), end_line) if end_line else len(lines)
        
        # 1-indexed to 0-indexed slice
        selected_lines = lines[start - 1 : end]
        content = "".join(selected_lines)
        
        return f"--- File: {filepath} (Lines {start}-{end}) ---\n{content}"
    except Exception as e:
        return f"Error reading file '{filepath}': {e!s}"

@tool
def search_web(query: str) -> str:
    """
    Search the web for up-to-date documentation, articles, or news.
    Useful for checking the latest framework syntax or solving modern issues.
    
    Args:
        query: The search query string.
    
    Returns:
        The search results snippet.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return f"No results found for '{query}'."
            
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"

# Export list of tools
AVAILABLE_TOOLS = [search_codebase, read_file, search_web]
