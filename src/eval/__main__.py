"""CLI entry point for eval.

Usage:
    python -m eval SESSION... [--output DIR]
    python -m eval --all [--output DIR]

Examples:
    python -m eval .msa/sessions/session-2026-05-25-005
    python -m eval session-2026-05-25-005 session-2026-05-25-007
    python -m eval --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval import eval_session, DEFAULT_EVALS_ROOT


def _resolve_session(arg: str) -> Path:
    path = Path(arg)
    if path.is_dir() and (path / "session.json").exists():
        return path
    # Try as session name under .msa/sessions/
    default = Path(".msa/sessions") / arg
    if default.is_dir() and (default / "session.json").exists():
        return default
    raise SystemExit(f"Not a session: {arg}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run eval checks on Mini Search Agent sessions.",
    )
    parser.add_argument(
        "sessions", nargs="*",
        help="Session paths or names under .msa/sessions/",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Eval all sessions under .msa/sessions/",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_EVALS_ROOT,
        help=f"Eval output root (default: {DEFAULT_EVALS_ROOT})",
    )
    args = parser.parse_args(argv)

    if args.all:
        sessions_dir = Path(".msa/sessions")
        if not sessions_dir.is_dir():
            raise SystemExit(".msa/sessions/ not found")
        targets = sorted(
            d for d in sessions_dir.iterdir()
            if d.is_dir() and (d / "session.json").exists()
        )
        if not targets:
            raise SystemExit("No sessions found")
    elif args.sessions:
        targets = [_resolve_session(s) for s in args.sessions]
    else:
        parser.print_help()
        raise SystemExit(1)

    for session_path in targets:
        print(f"Eval {session_path.name} ...", end=" ", flush=True)
        try:
            results = eval_session(
                session_path,
                evals_root=args.output,
            )
            fails = [r for r in results if r.score == 0]
            total = len(results)
            print(f"{total - len(fails)}/{total} pass" + (f" ({len(fails)} fail)" if fails else ""))
        except Exception as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
