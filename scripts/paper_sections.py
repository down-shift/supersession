#!/usr/bin/env python3
"""Split a paper Markdown file into ordered sections, or concatenate them."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER = ROOT / "paper" / "paper_draft.md"
DEFAULT_SECTIONS = ROOT / "paper" / "sections"
TOP_LEVEL_HEADING = re.compile(r"^## (?!#)(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
INDEX_LINK = re.compile(r"^- \[(.+?)\]\(([^)]+\.md)\)\s*$", re.MULTILINE)


def slugify(title: str) -> str:
    """Make a readable filename component from a Markdown heading."""
    title = title.lower().replace("×", " ")
    title = re.sub(r"^\d+\.\s*", "", title)
    title = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    return title or "section"


def top_level_headings(text: str) -> list[tuple[int, str]]:
    """Find level-two headings outside fenced code blocks."""
    found: list[tuple[int, str]] = []
    fence_char: str | None = None
    fence_length = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        fence = FENCE.match(content)
        if fence_char:
            if fence:
                token = fence.group(1)
                fence_closes = not content[len(token) :].strip()
                if fence_closes and token[0] == fence_char and len(token) >= fence_length:
                    fence_char = None
                    fence_length = 0
        elif fence:
            token = fence.group(1)
            fence_char, fence_length = token[0], len(token)
        else:
            heading = TOP_LEVEL_HEADING.match(content)
            if heading:
                found.append((offset, heading.group(1)))
        offset += len(line)
    return found


def existing_heading_names(sections_dir: Path) -> dict[str, str]:
    """Retain established filenames when a heading already has a section file."""
    names: dict[str, str] = {}
    if not sections_dir.is_dir():
        return names
    for path in sections_dir.glob("[0-9][0-9]-*.md"):
        text = path.read_text(encoding="utf-8")
        headings = top_level_headings(text)
        if headings:
            names[headings[0][1]] = path.name
    return names


def split_paper(paper_path: Path, sections_dir: Path) -> list[Path]:
    text = paper_path.read_text(encoding="utf-8")
    headings = top_level_headings(text)
    if not headings:
        raise ValueError(f"No level-two section headings found in {paper_path}")

    chunks: list[tuple[str, str]] = []
    preamble = text[: headings[0][0]].strip()
    if preamble:
        chunks.append(("", preamble))
    for index, (start, heading) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        chunks.append((heading, text[start:end].strip()))

    previous_names = existing_heading_names(sections_dir)
    paths: list[Path] = []
    used_names: set[str] = set()
    for index, (heading, content) in enumerate(chunks):
        if not heading:
            name = "00-front-matter.md"
        else:
            name = previous_names.get(heading)
            if name is None:
                name = f"{index:02d}-{slugify(heading)}.md"
        if name in used_names:
            raise ValueError(f"Two paper sections map to the same filename: {name}")
        used_names.add(name)
        path = sections_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
        paths.append(path)

    source_ref = Path(os.path.relpath(paper_path.resolve(), sections_dir.resolve())).as_posix()
    lines = [
        "# Paper sections",
        "",
        f"Split from `{source_ref}`. Sections are in manuscript order.",
        "",
    ]
    for path, (heading, _) in zip(paths, chunks):
        label = heading or "Front matter"
        if heading and re.match(r"^\d+\.\s", heading):
            label = re.sub(r"^(\d+)\.\s*", r"\1. ", heading)
        elif heading == "Abstract":
            label = "Abstract"
        elif heading == "References":
            label = "References"
        lines.append(f"- [{label}]({path.name})")
    (sections_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def ordered_section_paths(sections_dir: Path) -> list[Path]:
    index_path = sections_dir / "README.md"
    if not index_path.is_file():
        raise FileNotFoundError(f"Section index not found: {index_path}; run split first")
    links = INDEX_LINK.findall(index_path.read_text(encoding="utf-8"))
    if not links:
        raise ValueError(f"No section links found in {index_path}")

    paths: list[Path] = []
    for _, name in links:
        path = (sections_dir / name).resolve()
        if path.parent != sections_dir.resolve():
            raise ValueError(f"Section link must name a file in {sections_dir}: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Section file listed in README is missing: {path}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate section links in {index_path}")
    return paths


def concatenate_sections(sections_dir: Path, output_path: Path) -> None:
    paths = ordered_section_paths(sections_dir)
    sections = [path.read_text(encoding="utf-8").strip() for path in paths]
    if any(not section for section in sections):
        raise ValueError("A section file is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("split", help="split a paper into Markdown section files")
    split_parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    split_parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)

    concat_parser = commands.add_parser("concat", help="concatenate section files into one paper")
    concat_parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    concat_parser.add_argument("--output", type=Path, default=DEFAULT_PAPER)

    args = parser.parse_args()
    try:
        if args.command == "split":
            paths = split_paper(args.paper, args.sections)
            print(f"Wrote {len(paths)} sections to {args.sections}")
        else:
            concatenate_sections(args.sections, args.output)
            print(f"Wrote combined paper to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
