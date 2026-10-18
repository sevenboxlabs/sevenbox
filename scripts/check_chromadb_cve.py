"""
Single-purpose check: has CVE-2026-45829 (GHSA-f4j7-r4q5-qw2c, CVSS 9.3
pre-auth RCE in chromadb's HTTP server) been patched yet?

Why this exists, not just a note to "re-check periodically": a critical,
unpatched CVE in a transitive dependency (chromadb, pulled in by crewai's
Knowledge feature -- see docs/security-review.md section 1) isn't
something to document once and hope someone remembers. This makes "did it
get fixed" a single command with an unambiguous answer, not a manual
pip-audit read someone has to remember to interpret correctly.

Usage:
    python scripts/check_chromadb_cve.py

Exit code 0 if a fix now exists (patch shipped -- upgrade chromadb and
remove/relax the guard in sevenbox_adapters/crewai_adapter.py's
_reject_if_chromadb_backed_features_enabled), 1 if still unpatched or
chromadb isn't installed to check against, 2 on a real error running the
check itself (network/pip-audit failure) -- distinguished so a caller
can tell "confirmed still vulnerable" apart from "couldn't tell."
"""
from __future__ import annotations

import json
import subprocess
import sys

CVE_ID = "CVE-2026-45829"
GHSA_ID = "GHSA-f4j7-r4q5-qw2c"


def _run_pip_audit_json():
    proc = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", "--format", "json"],
        capture_output=True, text=True,
    )
    # pip-audit exits non-zero whenever it finds ANY vulnerability, in any
    # package -- that's expected here, not a failure of this check itself.
    return json.loads(proc.stdout), proc.stderr or proc.stdout


def main() -> int:
    installed_here = False
    try:
        import pip_audit  # noqa: F401
    except ImportError:
        # pip-audit's own dependency on tomli>=2.2.1 directly conflicts
        # with crewai's tomli~=2.0.2 pin -- there is no version of tomli
        # that satisfies both, verified while writing this script. So
        # this install is deliberately temporary: installed only for this
        # run, uninstalled in `finally` below, with tomli/tomli-w
        # restored to what crewai wants -- same cleanup this project's
        # own security-review session did by hand (docs/security-
        # review.md, "Environment hygiene during this scan"). Don't
        # remove this cleanup thinking it's unnecessary caution -- it
        # isn't, it was verified to matter.
        print("pip-audit not installed -- installing it temporarily for this check.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pip-audit"], check=True)
        installed_here = True

    try:
        data, raw_output = _run_pip_audit_json()
    except json.JSONDecodeError:
        print("Could not parse pip-audit output -- something else went wrong:")
        print(raw_output if installed_here else "(re-run with pip-audit already installed to see output)")
        return 2
    finally:
        if installed_here:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pip-audit"],
                            capture_output=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                             "tomli~=2.0.2", "tomli-w~=1.1.0"], capture_output=True)

    chromadb_entries = [d for d in data.get("dependencies", []) if d.get("name") == "chromadb"]
    if not chromadb_entries:
        print(
            "chromadb isn't installed in this environment, so this check can't "
            "tell you anything about its patch status. Run this from the venv "
            "that actually has crewai (and therefore chromadb) installed."
        )
        return 1

    for dep in chromadb_entries:
        version = dep.get("version", "?")
        matching = [v for v in dep.get("vulns", []) if v["id"] == "PYSEC-2026-311" or GHSA_ID in v.get("aliases", [])]
        if not matching:
            print(
                f"chromadb {version} installed here -- pip-audit does not flag it for "
                f"{CVE_ID} at all. Either already patched in this version, or the "
                "advisory data pip-audit is using is out of date -- cross-check "
                f"https://github.com/advisories/{GHSA_ID} directly before trusting this."
            )
            return 0
        for v in matching:
            fix_versions = v.get("fix_versions") or []
            if fix_versions:
                print(
                    f"PATCH AVAILABLE: {CVE_ID} is now fixed in chromadb "
                    f"{', '.join(fix_versions)} (installed here: {version}). "
                    "Upgrade, then remove or relax the guard in "
                    "sevenbox_adapters/crewai_adapter.py's "
                    "_reject_if_chromadb_backed_features_enabled -- see "
                    "docs/security-review.md section 1."
                )
                return 0
            print(
                f"Still unpatched as of this check: {CVE_ID} / {GHSA_ID}, "
                f"chromadb {version}, no fix_versions listed. Guard in "
                "sevenbox_adapters/crewai_adapter.py stays in place."
            )
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
