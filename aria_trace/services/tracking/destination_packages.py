"""Explicitly install prepared destination banks as reusable workbench artifacts."""
import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from .destination import DEFAULT_MASK, MODEL, MODEL_SHA256, SCHEMA, checked_id, load_package, read_json


def import_bank(bank_path, views_path, target, output_root, game_id, destination_id, label, source_description):
    checked_id(game_id); checked_id(destination_id)
    if not label.strip() or len(label) > 100:
        raise ValueError("Supply a destination label of at most 100 characters")
    views = read_json(views_path)[target]
    parent = Path(output_root) / game_id
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / destination_id
    if destination.exists():
        raise ValueError("Destination already exists; choose a new ID")
    # TemporaryDirectory uses a private Windows DACL. A published user artifact
    # must inherit the destination workspace's normal permissions instead.
    stage = parent / (".destination-import-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        with np.load(str(bank_path), allow_pickle=False) as source:
            arrays = {view["key"] + suffix: source[view["key"] + suffix] for view in views for suffix in ["p", "d", "o"]}
        np.savez_compressed(str(stage / "bank.npz"), **arrays)
        meta = dict(schema=SCHEMA, destination_id=destination_id, label=label.strip(), game_profile_id=game_id, model=MODEL, model_sha256=MODEL_SHA256, image_size_wh=[640, 360], mask=DEFAULT_MASK, view_count=len(views), views=views, source_description=source_description, preparation="Imported linked physical-goal views; no neural training", bank_sha256=hashlib.sha256((stage / "bank.npz").read_bytes()).hexdigest())
        (stage / "destination.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
        load_package(stage, game_id)
        # Publish only a complete validated directory; no partially ready catalog entry.
        stage.rename(destination)
    finally:
        if stage.exists():
            if stage.resolve().parent != parent.resolve() or not stage.name.startswith(".destination-import-"):
                raise RuntimeError("Refuse cleanup outside destination package staging")
            shutil.rmtree(str(stage))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-description", required=True)
    args = parser.parse_args()
    print(import_bank(args.bank, args.views, args.target, args.output_root, args.game, args.id, args.label, args.source_description))
