"""Verify the interpreter and direct dependencies without requiring data."""

import importlib
import importlib.metadata
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit("Use Python 3.11 with uv sync")
for module, package in [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("streamlit", "streamlit"),
    ("pyarrow", "pyarrow"),
]:
    importlib.import_module(module)
    print(f"OK {package} {importlib.metadata.version(package)}")
print(f"Python: {sys.version.split()[0]}")
