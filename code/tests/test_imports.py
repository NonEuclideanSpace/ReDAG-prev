"""Every module under redag/ and every script compiles and the library modules import (catches syntax errors in
code paths the mock tests never reach, such as calibrate.py's gate writer)."""
import importlib
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)


def test_imports():
    for sub in ("redag", "scripts"):
        for fn in sorted(os.listdir(os.path.join(CODE, sub))):
            if fn.endswith(".py"):
                py_compile.compile(os.path.join(CODE, sub, fn), doraise=True)
    for mod in ("redag.calibrate", "redag.labels", "redag.planner", "redag.schema", "redag.vlm", "redag.verify", "redag.diagnose", "redag.loop", "redag.report", "redag.preflight"):
        importlib.import_module(mod)


if __name__ == "__main__":
    test_imports()
    print("import/compile sweep ok")
