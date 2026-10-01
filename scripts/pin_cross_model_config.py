#!/usr/bin/env python3
"""Resolve a gated Hugging Face model to an immutable config revision."""
import logging
import re
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cross_model.pin")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/pin_cross_model_config.py BASE.yaml OUTPUT.yaml")
    base, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(base.read_text())
    model_id = config["model"]["id"]
    logger.info("Resolving immutable Hugging Face revision for %s", model_id)
    from huggingface_hub import model_info
    revision = model_info(model_id, revision="main").sha
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise RuntimeError(f"Hugging Face returned a non-immutable revision: {revision!r}")
    config["model"]["revision"] = revision
    config["model"]["tokenizer_revision"] = revision
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    logger.info("Pinned %s to %s; wrote %s", model_id, revision, output)


if __name__ == "__main__":
    main()
