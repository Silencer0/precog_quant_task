from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path, *, timeout_s: int) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout_s,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Notebook paths to execute")
    ap.add_argument(
        "--timeout", type=int, default=1800, help="Per-notebook timeout in seconds"
    )
    args = ap.parse_args()

    for raw in args.paths:
        p = Path(raw).resolve()
        if not p.exists():
            raise SystemExit(f"Notebook not found: {p}")
        print(f"EXECUTE {p}")
        execute_notebook(p, timeout_s=int(args.timeout))
        print(f"OK      {p}")


if __name__ == "__main__":
    main()
