"""Explicit publication inventory, excluding the downloaded book."""
from prepare import ROOT

ROOT_FILES = (".gitignore", "README.md", "USAGE.md", "LICENSE", "requirements.txt", "environment.lock.txt",
              "pyproject.toml", "repository.json", "prepare.py", "study.py", "benchmark.py", "diagnostics.py",
              "inspect_routes.py", "generate.py", "plots.py", "verify.py", "artifacts.py")


def files(include_audit=False):
    result = [ROOT/name for name in ROOT_FILES]+[ROOT/"data/manifest.json"]
    for folder, pattern in (("routeforge", "*.py"), ("tests", "*.py"), ("figures", "*.png")):
        result += sorted((ROOT/folder).glob(pattern))
    result += sorted(p for p in (ROOT/"runs").rglob("*") if p.is_file())
    if include_audit:
        result.append(ROOT/"audit.json")
    assert all(p.is_file() for p in result)
    return sorted(set(result))
