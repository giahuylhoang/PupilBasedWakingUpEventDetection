"""Test-session hygiene shared by every test module under vasomotion_app/tests/.

`streamlit.testing.v1.app_test` creates one module-level `tempfile.
TemporaryDirectory()` (its `TMP_DIR`) the moment it is imported, and never
calls `.cleanup()` on it itself -- it relies on that object's own finalizer,
which runs during interpreter shutdown (well after pytest has already
printed its summary) and calls `warnings.warn(..., ResourceWarning)` on the
"implicit" cleanup path. Under `-W error` that turns into a traceback printed
by Python's own atexit/weakref machinery -- harmless (pytest's exit code and
warning summary are unaffected; nothing here ever runs during a test) but
avoidable: cleaning it up explicitly, once, after the whole session finishes
takes the "implicit" path out of play entirely, so nothing warns in the
first place.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_streamlit_apptest_tmp_dir():
    yield
    try:
        from streamlit.testing.v1.app_test import TMP_DIR
    except ImportError:
        return
    TMP_DIR.cleanup()
