#!/usr/bin/env python3
"""Assemble the anonymous artifact package for submission (reviewer item E3).

Copies frozen design records, datasets, candidate/geometry audits, score files (which contain the
rendered prompts), analysis JSONs, provenance records (model revisions, generation and parser
settings), configs, the scoring code and uv.lock into outputs/artifact_package/ and writes
MANIFEST.tsv with one line per file:

    package_path  sha256_in_package  sha256_of_original  redacted

Sealed records written on the experiment machines contain absolute home-directory paths, host names and
machine aliases. These are replaced by neutral labels ("/home/anonymous", "machine-A/B/C") in the packaged
copy, so a redacted file's package hash
differs from the original hash printed in the paper; the manifest gives both, and the original
can be verified against the paper's hash after review. Nothing under outputs/ is modified.

Finally every SHA-256 printed in the paper and supplement is looked up in the manifest; the script
fails if one is missing.

    python scripts/package_artifacts.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/artifact_package"
HOME = re.compile(rb"/(home|Users)/[A-Za-z0-9_.-]+")
# Host names, storage paths and machine aliases that identify people or institutions.
NAMES = ((re.compile(rb"danya-Z790-D-AX"), b"machine-A"), (re.compile(rb"gpubox"), b"machine-A"),
         (re.compile(rb"NSSLabPC|nsspc"), b"machine-B"), (re.compile(rb"d\.dgx"), b"machine-C"),
         (re.compile(rb"kaluzhnaya[_A-Za-z]*"), b"anonymous"), (re.compile(rb"/data/storage/[A-Za-z0-9_]+"), b"/data/storage/anonymous"))
PATTERNS = (
    "outputs/**/*.jsonl", "outputs/**/*_analysis.json", "outputs/**/*analysis*.json", "outputs/**/*report*.json",
    "outputs/**/*.provenance.json", "outputs/**/*.run.json", "outputs/**/design_record*.json",
    "outputs/**/tokenizer_audit*.json", "outputs/**/vocabulary_audit*.json", "outputs/**/depth_selection.json", "outputs/**/*.geometry.json", "outputs/**/donor_following_by_order.json",
    "configs/**/*.yaml", "configs/**/*.json", "docs/*.md", "src/**/*.py", "scripts/*.py", "scripts/*.sh",
    "paper/data/*.py", "paper/data/*.csv", "paper/data/*.json", "paper/figures/*.py", "uv.lock", "pyproject.toml",
)
EXCLUDE = ("outputs/artifact_package/", "aborted_", "/migration_source/", "docs/PLAN.md", "docs/AUTONOMY.md")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    rows, seen = [], set()
    for pattern in PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            rel = path.relative_to(ROOT).as_posix()
            if rel in seen or not path.is_file() or any(x in rel for x in EXCLUDE):
                continue
            seen.add(rel)
            data = path.read_bytes()
            packaged = HOME.sub(b"/home/anonymous", data)
            for pattern, repl in NAMES:
                packaged = pattern.sub(repl, packaged)
            target = OUT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(packaged)
            rows.append((rel, sha(packaged), sha(data), "yes" if packaged != data else "no"))
    (OUT / "MANIFEST.tsv").write_text(
        "package_path\tsha256_in_package\tsha256_of_original\tredacted\n"
        + "".join("\t".join(r) + "\n" for r in rows))
    printed = set()
    for tex in (ROOT / "paper/supplement.tex", *sorted((ROOT / "paper/sections").glob("*.tex"))):
        printed |= set(re.findall(r"\b[0-9a-f]{64}\b", tex.read_text()))
    known = {r[2] for r in rows} | {r[1] for r in rows}
    # A printed hash may instead be a digest recorded inside a packaged record (e.g. a code digest).
    recorded = set()
    for r in rows:
        if r[0].endswith(".json"):
            recorded |= set(re.findall(r"\b[0-9a-f]{64}\b", (OUT / r[0]).read_text(errors="ignore")))
    as_record = sorted((printed - known) & recorded)
    missing = sorted(printed - known - recorded)
    redacted = sum(r[3] == "yes" for r in rows)
    (OUT / "RECORDED_DIGESTS.txt").write_text(
        "Hashes printed in the paper that are digests recorded inside packaged records, not file hashes:\n"
        + "".join(h + "\n" for h in as_record))
    print(f"{len(rows)} files, {redacted} redacted, {len(printed)} printed hashes: "
          f"{len(printed) - len(as_record) - len(missing)} file hashes, {len(as_record)} recorded digests, "
          f"{len(missing)} not found")
    for h in missing:
        print("  missing:", h)
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
