"""
sevenbox_demo.cli -- the `sevenbox` command.

`sevenbox ops` and `sevenbox demo` are the same command under two names, both
dispatching to the same `run_demo_server()`. `ops` is the one to reach
for and to document going forward -- it starts the Operations Center
(server up, browser open, on the same page every other Epic 2 commit has
been building), matching what the tool actually is today, not what it
was before Epic 2. `demo` keeps working unchanged: an additive alias,
never removed, per this project's own backward-compatibility discipline
-- the same posture README.md's language audit
already flagged for the deeper `sevenbox_demo` package/command rename it
deliberately left out of scope. That structural rename stays out of
scope here too; this only adds a second, equally real name for the same
entry point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_server_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--port", type=int, default=8420, help="Port to listen on (default: 8420).")
    subparser.add_argument(
        "--no-browser", action="store_true", help="Don't open a browser tab automatically."
    )


def _cmd_init(filename: str) -> int:
    """Writes a real, runnable starter script -- the exact pattern an
    external integrator would hand-write (sevenbox_sdk decorators +
    SqliteWarrantStore), generated instead of requiring anyone to read
    the SDK before producing their first Warrant. Refuses to overwrite an
    existing file rather than silently clobbering whatever's there.
    """
    from .init_scaffold import SCAFFOLD_TEMPLATE

    target = Path.cwd() / filename
    if target.exists():
        print(f"{filename} already exists -- not overwriting.")
        print(f"Pick a different name: sevenbox init <name>.py")
        return 1

    target.write_text(SCAFFOLD_TEMPLATE, encoding="utf-8")
    print(f"Created {filename}")
    print()
    print("Next steps:")
    print(f"  1. python {filename}")
    print("  2. sevenbox ops")
    print('  3. "Open System..." -> "Your own agent" -> sevenbox_store.db')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sevenbox", description="Sevenbox command-line tools.")
    subparsers = parser.add_subparsers(dest="command")

    # Registration order is the order `sevenbox --help` lists them in --
    # init first, matching the actual order a new user follows (generate
    # a script, then look at what it produced), demo last since it's the
    # legacy alias nobody needs to reach for on purpose anymore.
    init_parser = subparsers.add_parser(
        "init", help="Generate a starter script that instruments a toy agent with Sevenbox."
    )
    init_parser.add_argument(
        "filename", nargs="?", default="my_agent.py",
        help="Name of the file to create (default: my_agent.py).",
    )

    ops_parser = subparsers.add_parser(
        "ops", help="Start the Sevenbox Operations Center (server + browser)."
    )
    _add_server_args(ops_parser)

    demo_parser = subparsers.add_parser(
        "demo", help="Same as `sevenbox ops` -- kept for backward compatibility."
    )
    _add_server_args(demo_parser)

    args = parser.parse_args(argv)

    if args.command in ("ops", "demo"):
        from .server import run_demo_server
        run_demo_server(port=args.port, open_browser=not args.no_browser)
        return 0

    if args.command == "init":
        return _cmd_init(args.filename)

    parser.print_help()
    return 0 if argv is None else 1


if __name__ == "__main__":
    sys.exit(main())
