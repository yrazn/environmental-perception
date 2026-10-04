#!/usr/bin/env python3
"""Prepare Indoor Fire Smoke as a leakage-safe YOLO detection dataset.

The downloaded Roboflow export has three issues that matter to this project:

1. ``data.yaml`` contains a placeholder path.
2. Its classes are ``0=fire, 1=smoke``, while this project uses
   ``0=smoke, 1=fire``.
3. Augmented variants of the same source image occur in different original
   splits.  A random image-level split would therefore leak visual content.

This script keeps the source export untouched, groups images by the filename
before ``.rf.<hash>``, creates a deterministic 70/15/15 grouped split, remaps
the classes, clips boxes to image bounds and writes audit reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SOURCE_SPLITS = ("train", "valid", "test")
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
# Source export: 0=fire, 1=smoke. Project convention: 0=smoke, 1=fire.
CLASS_REMAP = {0: 1, 1: 0}
CLASS_NAMES = {0: "smoke", 1: "fire"}


@dataclass(frozen=True)
class Record:
    source_split: str
    image_path: Path
    label_path: Path
    source_id: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def source_id(path: Path) -> str:
    """Return the pre-augmentation identity embedded in Roboflow filenames."""
    return path.stem.split(".rf.", 1)[0]


def read_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(label_path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_value, x, y, width, height = map(float, fields)
        class_id = int(class_value)
        if class_value != class_id or class_id not in CLASS_REMAP:
            raise ValueError(
                f"{label_path}:{line_number}: unsupported class {class_value}"
            )
        if not all(map(lambda value: value == value, (x, y, width, height))):
            raise ValueError(f"{label_path}:{line_number}: NaN coordinate")
        rows.append((class_id, x, y, width, height))
    return rows


def classify(rows: list[tuple[int, float, float, float, float]]) -> str:
    classes = {row[0] for row in rows}
    if not classes:
        return "negative"
    if classes == {0}:
        return "fire_only"
    if classes == {1}:
        return "smoke_only"
    return "both"


def collect_records(source: Path) -> list[Record]:
    records: list[Record] = []
    for split in SOURCE_SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images/labels directory in {source / split}")

        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        labels = {
            path.stem: path
            for path in label_dir.glob("*.txt")
            if not path.name.startswith(".")
        }
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels or missing_images:
            raise ValueError(
                f"{split}: {len(missing_labels)} images without labels, "
                f"{len(missing_images)} labels without images"
            )

        for stem in sorted(images):
            rows = read_rows(labels[stem])
            records.append(
                Record(split, images[stem], labels[stem], source_id(images[stem]), classify(rows))
            )
    return records


def assign_grouped_splits(records: list[Record], seed: int) -> dict[str, str]:
    """Stratify source-image groups while keeping all augmentations together."""
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.source_id].append(record)

    strata: dict[str, list[str]] = defaultdict(list)
    for group_id, group_records in grouped.items():
        present = {item.category for item in group_records}
        if "both" in present or {"fire_only", "smoke_only"} <= present:
            category = "both"
        elif "fire_only" in present:
            category = "fire_only"
        elif "smoke_only" in present:
            category = "smoke_only"
        else:
            category = "negative"
        strata[category].append(group_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for category, group_ids in sorted(strata.items()):
        rng.shuffle(group_ids)
        # Large augmentation groups go first so the greedy allocation remains
        # close to the desired image-level ratio rather than just group ratio.
        group_ids.sort(key=lambda key: len(grouped[key]), reverse=True)
        totals = {split: 0 for split in TARGET_RATIOS}
        target = {
            split: sum(len(grouped[key]) for key in group_ids) * ratio
            for split, ratio in TARGET_RATIOS.items()
        }
        for group_id in group_ids:
            chosen = min(
                TARGET_RATIOS,
                key=lambda split: (
                    totals[split] / target[split] if target[split] else 0.0,
                    split,
                ),
            )
            assignment[group_id] = chosen
            totals[chosen] += len(grouped[group_id])
    return assignment


def clip_and_remap(
    rows: list[tuple[int, float, float, float, float]]
) -> tuple[list[str], int, int]:
    output: list[str] = []
    clipped = 0
    removed = 0
    for class_id, x, y, width, height in rows:
        left, top = x - width / 2, y - height / 2
        right, bottom = x + width / 2, y + height / 2
        clipped_box = (
            max(0.0, min(1.0, left)),
            max(0.0, min(1.0, top)),
            max(0.0, min(1.0, right)),
            max(0.0, min(1.0, bottom)),
        )
        if clipped_box != (left, top, right, bottom):
            clipped += 1
        left, top, right, bottom = clipped_box
        new_width, new_height = right - left, bottom - top
        if new_width <= 0 or new_height <= 0:
            removed += 1
            continue
        new_x, new_y = (left + right) / 2, (top + bottom) / 2
        output.append(
            f"{CLASS_REMAP[class_id]} {new_x:.8f} {new_y:.8f} "
            f"{new_width:.8f} {new_height:.8f}"
        )
    return output, clipped, removed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: Path, output: Path, seed: int) -> dict[str, object]:
    records = collect_records(source)
    assignment = assign_grouped_splits(records, seed)

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Remove or choose another output path."
        )

    for split in TARGET_RATIOS:
        (output / "data" / split / "images").mkdir(parents=True, exist_ok=True)
        (output / "data" / split / "labels").mkdir(parents=True, exist_ok=True)
    report_dir = output / "reports"
    report_dir.mkdir(parents=True)

    counters: Counter[str] = Counter()
    image_hashes: dict[str, str] = {}
    manifest_rows: list[dict[str, object]] = []

    for record in records:
        target_split = assignment[record.source_id]
        image_target = output / "data" / target_split / "images" / record.image_path.name
        label_target = output / "data" / target_split / "labels" / f"{record.image_path.stem}.txt"
        if image_target.exists() or label_target.exists():
            raise FileExistsError(f"Target filename collision: {record.image_path.name}")

        # Decode before copying so unreadable images fail preparation instead of
        # surfacing during an expensive training run.
        with Image.open(record.image_path) as image:
            image.verify()
        shutil.copy2(record.image_path, image_target)

        rows = read_rows(record.label_path)
        labels, clipped, removed = clip_and_remap(rows)
        label_target.write_text("\n".join(labels) + ("\n" if labels else ""))

        digest = sha256(record.image_path)
        if digest in image_hashes:
            counters["exact_duplicate_images"] += 1
        else:
            image_hashes[digest] = record.image_path.as_posix()

        counters[f"{target_split}_images"] += 1
        counters[f"{target_split}_{record.category}_images"] += 1
        counters[f"{target_split}_boxes"] += len(labels)
        counters["boxes_clipped"] += clipped
        counters["boxes_removed"] += removed
        for label in labels:
            counters[f"{target_split}_{CLASS_NAMES[int(label.split()[0])]}_boxes"] += 1

        manifest_rows.append(
            {
                "source_split": record.source_split,
                "target_split": target_split,
                "source_id": record.source_id,
                "image": record.image_path.name,
                "source_category": record.category,
                "boxes_before": len(rows),
                "boxes_after": len(labels),
                "clipped_boxes": clipped,
                "sha256": digest,
            }
        )

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        f"path: {output / 'data'}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )

    with (report_dir / "split_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Prove the key anti-leakage invariant in the generated report.
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        source_splits[str(row["source_id"])].add(str(row["target_split"]))
    leaking_groups = [key for key, splits in source_splits.items() if len(splits) > 1]

    summary: dict[str, object] = {
        "source": source.as_posix(),
        "output": output.as_posix(),
        "seed": seed,
        "source_class_mapping": {"0": "fire", "1": "smoke"},
        "target_class_mapping": {"0": "smoke", "1": "fire"},
        "split_method": "grouped by pre-.rf source id; stratified by class presence",
        "source_groups": len(source_splits),
        "cross_target_split_source_groups": len(leaking_groups),
        "counters": dict(sorted(counters.items())),
    }
    (report_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare(args.source.resolve(), args.output.resolve(), args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
