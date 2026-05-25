from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import ConfigError
from .runner import run_research


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mini-search-agent")
    parser.add_argument("question", help="Open Research Question to investigate")
    args = parser.parse_args(argv)

    try:
        run_research(args.question, output=sys.stdout)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
