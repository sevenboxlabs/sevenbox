"""
Session-wide test isolation for warrant_core's first-import terms gate
(warrant_core/_terms_gate.py). This must run at module level, not inside a
fixture: the very first `import warrant_core` anywhere under tests/ happens
during pytest's *collection* of that test module's top-level `from
warrant_core import ...` statement -- before any fixture has a chance to
run. conftest.py is loaded before collection starts, which is the only
point early enough.

Without this, the whole suite would either hang waiting for input() (if a
real TTY is attached) or fail every single test that imports warrant_core
(if not) the first time pytest runs in a fresh environment with no
~/.sevenbox/accepted_terms marker yet -- and even when a marker already
exists, running the suite would be writing into the real developer's home
directory as a side effect of running tests, which
tests/test_paper_version_sync.py-style discipline treats as exactly the
kind of thing a test must not do.
"""
import os
import tempfile

_TEST_SEVENBOX_CONFIG_DIR = tempfile.mkdtemp(prefix="sevenbox-test-config-")
os.environ["SEVENBOX_CONFIG_DIR"] = _TEST_SEVENBOX_CONFIG_DIR
os.environ["SEVENBOX_ACCEPT_TERMS"] = "1"
