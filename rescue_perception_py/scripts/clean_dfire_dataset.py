#!/usr/bin/env python3
"""清洗 D-Fire YOLO 数据，并生成降低连续帧泄漏风险的新划分。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "smoke", 1: "fire"}


@dataclass(frozen=True)
class Sample:
    """一张图像、标签及用于分组划分的元数据。"""

    stem: str
    image_path: Path
    label_path: Path
    source_split: str
    prefix: str
    number: int
    category: str


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "datasets/D-Fire/archive",
        help="原始 D-Fire 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "datasets/D-Fire/clean",
        help="清洗后数据集根目录",
    )
    parser.add_argument("--group-size", type=int, default=250)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--buffer-frames", type=int, default=5)
    parser.add_argument("--search-seeds", type=int, default=3000)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="复制图像而不是创建硬链接（会额外占用约3 GB）",
    )
    return parser.parse_args()


def label_category(label_path: Path) -> str:
    """按标签中出现的类别将图像归为负样本、单类或双类样本。"""
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            classes.add(int(parts[0]))
        except ValueError:
            continue
    if not classes:
        return "negative"
    if classes == {0}:
        return "smoke_only"
    if classes == {1}:
        return "fire_only"
    return "both"


def load_samples(source_root: Path, splits: Iterable[str]) -> list[Sample]:
    """读取图像与同名标签，并从文件名解析来源前缀和数字编号。"""
    samples: list[Sample] = []
    pattern = re.compile(r"^(.*?)(\d+)$")
    for split in splits:
        image_dir = source_root / "data" / split / "images"
        label_dir = source_root / "data" / split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"缺少标签: {label_path}")
            match = pattern.match(image_path.stem)
            if not match:
                raise ValueError(f"无法解析连续编号文件名: {image_path.name}")
            samples.append(
                Sample(
                    stem=image_path.stem,
                    image_path=image_path,
                    label_path=label_path,
                    source_split=split,
                    prefix=match.group(1),
                    number=int(match.group(2)),
                    category=label_category(label_path),
                )
            )
    return samples


def choose_validation_groups(
    samples: list[Sample], group_size: int, val_ratio: float, search_seeds: int
) -> tuple[set[tuple[str, int]], int, float]:
    """搜索兼顾样本量和四种图像类别比例的分组验证集。"""
    groups: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for sample in samples:
        groups[(sample.prefix, sample.number // group_size)][sample.category] += 1

    group_keys = sorted(groups)
    total = Counter(sample.category for sample in samples)
    target_count = len(samples) * val_ratio
    categories = ("negative", "smoke_only", "fire_only", "both")
    best: tuple[float, int, int, set[tuple[str, int]]] | None = None

    for seed in range(search_seeds):
        rng = random.Random(seed)
        shuffled = group_keys.copy()
        rng.shuffle(shuffled)
        selected: set[tuple[str, int]] = set()
        selected_count = 0
        selected_categories: Counter[str] = Counter()
        for key in shuffled:
            if selected_count >= target_count:
                break
            selected.add(key)
            selected_categories.update(groups[key])
            selected_count += sum(groups[key].values())

        size_error = abs(selected_count / len(samples) - val_ratio)
        distribution_error = sum(
            abs(
                selected_categories[category] / selected_count
                - total[category] / len(samples)
            )
            for category in categories
        )
        score = size_error * 5.0 + distribution_error
        candidate = (score, seed, selected_count, selected)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    assert best is not None
    score, seed, _, selected = best
    return selected, seed, score


def assign_splits(
    development_samples: list[Sample],
    validation_groups: set[tuple[str, int]],
    group_size: int,
    buffer_frames: int,
) -> tuple[dict[str, str], set[str]]:
    """按连续编号块划分 train/val，并从 train 侧移除边界缓冲帧。"""
    assignments: dict[str, str] = {}
    validation_numbers: dict[str, set[int]] = defaultdict(set)

    for sample in development_samples:
        group = (sample.prefix, sample.number // group_size)
        split = "val" if group in validation_groups else "train"
        assignments[sample.stem] = split
        if split == "val":
            validation_numbers[sample.prefix].add(sample.number)

    excluded: set[str] = set()
    for sample in development_samples:
        if assignments[sample.stem] != "train":
            continue
        val_numbers = validation_numbers[sample.prefix]
        if any(
            sample.number + offset in val_numbers
            for offset in range(-buffer_frames, buffer_frames + 1)
        ):
            excluded.add(sample.stem)
    return assignments, excluded


def clean_label(
    sample: Sample,
    destination: Path,
    review_rows: list[dict[str, object]],
) -> Counter[str]:
    """删除非法/零尺寸框，并将部分越界框裁剪回归一化图像范围。"""
    stats: Counter[str] = Counter()
    cleaned_lines: list[str] = []
    for line_number, line in enumerate(
        sample.label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        parts = line.split()
        if len(parts) != 5:
            stats["dropped_malformed"] += 1
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            stats["dropped_malformed"] += 1
            continue
        if class_id not in CLASS_NAMES:
            stats["dropped_unknown_class"] += 1
            continue

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        overshoot = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0, width - 1.0, height - 1.0)

        if width <= 0.0 or height <= 0.0:
            stats["dropped_nonpositive"] += 1
            review_rows.append(
                {
                    "split": sample.source_split,
                    "image": sample.image_path.name,
                    "label": sample.label_path.name,
                    "line": line_number,
                    "class_id": class_id,
                    "reason": "nonpositive_size",
                    "overshoot": f"{overshoot:.8f}",
                    "original": line,
                }
            )
            continue

        clipped_x1 = min(1.0, max(0.0, x1))
        clipped_y1 = min(1.0, max(0.0, y1))
        clipped_x2 = min(1.0, max(0.0, x2))
        clipped_y2 = min(1.0, max(0.0, y2))
        clipped_width = clipped_x2 - clipped_x1
        clipped_height = clipped_y2 - clipped_y1
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            stats["dropped_after_clip"] += 1
            continue

        if overshoot > 1e-6:
            stats["clipped_boxes"] += 1
            if overshoot > 0.01:
                stats["review_boxes"] += 1
                review_rows.append(
                    {
                        "split": sample.source_split,
                        "image": sample.image_path.name,
                        "label": sample.label_path.name,
                        "line": line_number,
                        "class_id": class_id,
                        "reason": "overshoot_gt_0.01",
                        "overshoot": f"{overshoot:.8f}",
                        "original": line,
                    }
                )

        new_x = (clipped_x1 + clipped_x2) / 2.0
        new_y = (clipped_y1 + clipped_y2) / 2.0
        cleaned_lines.append(
            f"{class_id} {new_x:.8f} {new_y:.8f} {clipped_width:.8f} {clipped_height:.8f}"
        )
        stats["kept_boxes"] += 1

    destination.write_text(
        "\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8"
    )
    return stats


def place_image(source: Path, destination: Path, copy_images: bool) -> str:
    """默认用硬链接节省空间，跨文件系统时自动回退为复制。"""
    if copy_images:
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖请先检查: {output_root}")
    if args.group_size <= 0 or args.buffer_frames < 0 or not 0 < args.val_ratio < 1:
        raise ValueError("group-size、buffer-frames 或 val-ratio 参数不合法")

    development_samples = load_samples(source_root, ("train", "val"))
    test_samples = load_samples(source_root, ("test",))
    validation_groups, selected_seed, split_score = choose_validation_groups(
        development_samples, args.group_size, args.val_ratio, args.search_seeds
    )
    assignments, excluded = assign_splits(
        development_samples, validation_groups, args.group_size, args.buffer_frames
    )

    for split in ("train", "val", "test"):
        (output_root / "data" / split / "images").mkdir(parents=True, exist_ok=False)
        (output_root / "data" / split / "labels").mkdir(parents=True, exist_ok=False)
    reports_dir = output_root / "reports"
    reports_dir.mkdir()

    review_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    global_stats: Counter[str] = Counter()
    split_stats: dict[str, Counter[str]] = defaultdict(Counter)
    link_modes: Counter[str] = Counter()

    all_samples = development_samples + test_samples
    for sample in all_samples:
        if sample.source_split == "test":
            destination_split = "test"
        else:
            destination_split = assignments[sample.stem]
        is_excluded = sample.stem in excluded
        manifest_rows.append(
            {
                "stem": sample.stem,
                "source_split": sample.source_split,
                "clean_split": "excluded_buffer" if is_excluded else destination_split,
                "prefix": sample.prefix,
                "number": sample.number,
                "group": sample.number // args.group_size,
                "category": sample.category,
            }
        )
        if is_excluded:
            split_stats["excluded_buffer"]["images"] += 1
            continue

        image_destination = (
            output_root / "data" / destination_split / "images" / sample.image_path.name
        )
        label_destination = (
            output_root / "data" / destination_split / "labels" / f"{sample.stem}.txt"
        )
        link_modes[place_image(sample.image_path, image_destination, args.copy_images)] += 1
        label_stats = clean_label(sample, label_destination, review_rows)
        global_stats.update(label_stats)
        split_stats[destination_split].update(label_stats)
        split_stats[destination_split]["images"] += 1
        split_stats[destination_split][sample.category] += 1

    with (reports_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (reports_dir / "review_boxes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["split", "image", "label", "line", "class_id", "reason", "overshoot", "original"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    data_yaml = (
        f"path: {output_root}\n"
        "train: data/train/images\n"
        "val: data/val/images\n"
        "test: data/test/images\n\n"
        "names:\n"
        "  0: smoke\n"
        "  1: fire\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    summary = {
        "source": str(source_root),
        "output": str(output_root),
        "split_method": {
            "group_size": args.group_size,
            "target_validation_ratio": args.val_ratio,
            "selected_seed": selected_seed,
            "selection_score": split_score,
            "buffer_frames_removed_from_train": args.buffer_frames,
            "validation_group_count": len(validation_groups),
        },
        "source_images": len(all_samples),
        "output_images": sum(stats["images"] for key, stats in split_stats.items() if key != "excluded_buffer"),
        "link_modes": dict(link_modes),
        "label_cleaning": dict(global_stats),
        "split_statistics": {key: dict(value) for key, value in split_stats.items()},
        "review_row_count": len(review_rows),
    }
    (reports_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# D-Fire clean dataset\n\n"
        "该目录由 `scripts/clean_dfire_dataset.py` 生成，原始 `archive` 未修改。\n\n"
        "- 类别：`0 smoke`、`1 fire`。\n"
        "- 非法或零尺寸框被删除，越界框被裁剪到 `[0, 1]`。\n"
        f"- 原 train/val 按每 {args.group_size} 个连续编号组成一组重新划分。\n"
        f"- train 侧与 val 相距 {args.buffer_frames} 个编号以内的边界样本被排除。\n"
        "- 原 test 保持不变，仅清洗标签。\n"
        "- `reports/split_manifest.csv` 记录每张图像的去向。\n"
        "- `reports/review_boxes.csv` 记录严重越界和零尺寸框。\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
