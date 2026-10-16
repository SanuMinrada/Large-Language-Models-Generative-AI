#!/usr/bin/env python3
"""Create today's (or a back-dated) daily log entry and link it from README.md."""

import argparse
import datetime
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
TEMPLATE_PATH = ROOT / "templates" / "daily-log.md"
README_PATH = ROOT / "README.md"
LOG_START = "<!-- LOG-START -->"
LOG_END = "<!-- LOG-END -->"


def parse_args():
    parser = argparse.ArgumentParser(description="Create a new daily log entry.")
    parser.add_argument("--topic", default="", help="Topic to fill in the log's Topic line")
    parser.add_argument("--date", default=None, help="Back-dated entry in YYYY-MM-DD format")
    return parser.parse_args()


def resolve_date(date_arg):
    if date_arg is None:
        return datetime.date.today()
    return datetime.datetime.strptime(date_arg, "%Y-%m-%d").date()


def create_log_file(date, topic):
    log_path = LOGS_DIR / f"{date.isoformat()}.md"
    if log_path.exists():
        return log_path, False

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    content = template.replace("## Date\n", f"## Date\n{date.isoformat()}\n", 1)
    if topic:
        content = content.replace("## Topic\n", f"## Topic\n{topic}\n", 1)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")
    return log_path, True


def update_readme(date, topic):
    if not README_PATH.exists():
        return

    readme = README_PATH.read_text(encoding="utf-8")
    if LOG_START not in readme or LOG_END not in readme:
        return

    link_line = f"- [{date.isoformat()}](logs/{date.isoformat()}.md) — {topic}".rstrip(" —")

    start_idx = readme.index(LOG_START) + len(LOG_START)
    end_idx = readme.index(LOG_END)
    block = readme[start_idx:end_idx]

    existing_lines = [line for line in block.strip("\n").split("\n") if line.strip()]
    date_prefix = f"- [{date.isoformat()}]"
    existing_lines = [line for line in existing_lines if not line.startswith(date_prefix)]

    new_lines = [link_line] + existing_lines
    new_block = "\n" + "\n".join(new_lines) + "\n"

    readme = readme[:start_idx] + new_block + readme[end_idx:]
    README_PATH.write_text(readme, encoding="utf-8")


def main():
    args = parse_args()
    date = resolve_date(args.date)

    log_path, created = create_log_file(date, args.topic)
    update_readme(date, args.topic)

    if created:
        print(log_path)
    else:
        print(f"Log already exists: {log_path}")


if __name__ == "__main__":
    main()
