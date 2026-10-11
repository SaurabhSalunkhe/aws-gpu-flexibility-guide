"""Read-only scan CLI. Optional reports use an exclusive create outside the repo."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

from .scanner import Limits, assess, load_catalog, render_markdown, open_directory_no_symlinks


def positive(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from None
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return result


def write_report(output, root, text):
    """Never overwrite; reject all symlink parent components and repo destinations.

    Output is explicitly requested, new, local and private (0600). No mkdir or
    chmod of existing resources. The output pathname is never echoed on error.
    """
    output = Path(os.path.abspath(Path(output).expanduser()))
    check_fd = open_directory_no_symlinks(root)
    os.close(check_fd)
    root = Path(os.path.abspath(Path(root).expanduser()))
    if output == root or root in output.parents:
        raise ValueError("Report output must be outside the scanned repository.")
    parent_fd = os.open(output.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in output.parent.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    finally:
        os.close(parent_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline bounded static GPU migration triage. Never executes customer code. Does not certify portability.")
    parser.add_argument("repository", nargs="?", help="Existing local repository directory; read-only.")
    parser.add_argument("--source-instance", help="Exact source EC2 instance size from the bundled catalog.")
    parser.add_argument("--target-instance", help="Exact target EC2 instance size from the bundled catalog.")
    parser.add_argument("--workload", choices=["unknown", "serving", "offline-inference", "training"], default="unknown")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", help="New report file OUTSIDE the repository; existing files and symlink paths are refused. Default: stdout.")
    parser.add_argument("--list-instances", action="store_true", help="List the exact sizes in the offline catalog and exit.")
    parser.add_argument("--max-files", type=positive, default=Limits.max_files)
    parser.add_argument("--max-file-bytes", type=positive, default=Limits.max_file_bytes)
    parser.add_argument("--max-total-bytes", type=positive, default=Limits.max_total_bytes)
    parser.add_argument("--max-entries", type=positive, default=Limits.max_entries)
    parser.add_argument("--max-depth", type=positive, default=Limits.max_depth)
    parser.add_argument("--max-findings", type=positive, default=Limits.max_findings)
    args = parser.parse_args(argv)
    if args.list_instances:
        print("\n".join(row["instance_type"] for row in load_catalog()["instances"]))
        return 0
    if not args.repository:
        parser.error("repository is required unless --list-instances is used")
    try:
        limits = Limits(max_files=args.max_files, max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes, max_entries=args.max_entries,
            max_depth=args.max_depth, max_findings=args.max_findings)
        report = assess(args.repository, args.source_instance, args.target_instance, args.workload, limits)
        text = json.dumps(report, indent=2, ensure_ascii=True) + "\n" if args.format == "json" else render_markdown(report)
        if args.output:
            write_report(args.output, args.repository, text)
        else:
            print(text, end="")
        return 0
    except ValueError as exc:
        # Our ValueErrors contain only fixed messages, not customer paths/content.
        print("Assessment error: " + str(exc), file=sys.stderr)
        return 2
    except OSError:
        print("Assessment error: local file access failed; an output may already exist, use a symlink, or lack permission. No existing file was overwritten.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
