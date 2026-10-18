import subprocess
import sys

import sevenbox_demo.server as server_module
from sevenbox_demo.cli import main


def test_ops_and_demo_both_dispatch_to_run_demo_server(monkeypatch):
    """`sevenbox ops` and `sevenbox demo` are one command under two names -- both
    must call the exact same run_demo_server(), with the exact same
    argument handling, never two separate implementations to keep in sync.
    """
    calls = []
    monkeypatch.setattr(
        server_module, "run_demo_server",
        lambda port, open_browser: calls.append((port, open_browser)),
    )

    assert main(["ops"]) == 0
    assert main(["demo"]) == 0
    assert calls == [(8420, True), (8420, True)]


def test_ops_and_demo_accept_the_same_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(
        server_module, "run_demo_server",
        lambda port, open_browser: calls.append((port, open_browser)),
    )

    main(["ops", "--port", "9001", "--no-browser"])
    main(["demo", "--port", "9001", "--no-browser"])
    assert calls == [(9001, False), (9001, False)]


def test_no_command_prints_help_and_does_not_start_a_server(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(server_module, "run_demo_server", lambda **kw: started.append(kw))

    main([])  # explicit empty argv (unlike argv=None) is the existing "no command" error path
    assert not started
    assert "sevenbox" in capsys.readouterr().out


def test_init_creates_a_starter_script_with_the_default_name(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / "my_agent.py").is_file()
    out = capsys.readouterr().out
    assert "sevenbox ops" in out  # next-step guidance, not just silence


def test_init_accepts_a_custom_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "onboarding_test.py"]) == 0
    assert (tmp_path / "onboarding_test.py").is_file()
    assert not (tmp_path / "my_agent.py").exists()


def test_init_refuses_to_overwrite_an_existing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_agent.py").write_text("# something the user already wrote\n")

    exit_code = main(["init"])
    assert exit_code == 1
    assert (tmp_path / "my_agent.py").read_text() == "# something the user already wrote\n"
    assert "already exists" in capsys.readouterr().out


def test_init_generated_script_actually_runs_and_produces_a_loadable_store(tmp_path):
    """The real end-to-end proof: not just that a file gets written, but
    that running it -- exactly as a first-time user would -- produces a
    SqliteWarrantStore sevenbox ops can open. Runs the generated file as its
    own subprocess (a real `python my_agent.py`), not an in-process import,
    so this proves nothing was special-cased to the test harness.
    """
    main_argv_cwd = tmp_path
    result_init = subprocess.run(
        [sys.executable, "-m", "sevenbox_demo.cli", "init"],
        cwd=main_argv_cwd, capture_output=True, text=True,
    )
    assert result_init.returncode == 0, result_init.stderr

    result_run = subprocess.run(
        [sys.executable, "my_agent.py"], cwd=main_argv_cwd, capture_output=True, text=True,
    )
    assert result_run.returncode == 0, result_run.stderr
    assert (tmp_path / "sevenbox_store.db").is_file()

    from warrant_core import SqliteWarrantStore

    store = SqliteWarrantStore(str(tmp_path / "sevenbox_store.db"))
    try:
        warrants = store.all()
        assert len(warrants) == 4  # axiom + oracle + tool + derive, per the scaffold
        assert {w.source_kind.value for w in warrants} == {"axiom", "oracle", "tool", "derived"}
    finally:
        store.close()
