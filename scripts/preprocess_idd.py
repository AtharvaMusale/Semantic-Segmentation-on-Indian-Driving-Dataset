"""Build segmentation masks and metadata from IDD polygon annotations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from idd_config import LABEL_TO_MASK_VALUE


DEFAULT_DRIVE_ID = "1iQ93IWVdR6dZ6W7RahbLq166u-6ADelJ"


@dataclass(frozen=True)
class Annotation:
    width: int
    height: int
    labels: list[str]
    polygons: list[list[tuple[float, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grayscale semantic masks from IDD polygon JSON files."
    )
    parser.add_argument("--data-dir", default="data", type=Path, help="Dataset root.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download data.zip with gdown before preprocessing.",
    )
    parser.add_argument(
        "--drive-id",
        default=DEFAULT_DRIVE_ID,
        help="Google Drive file id used when --download is set.",
    )
    parser.add_argument(
        "--zip-path",
        default="data.zip",
        type=Path,
        help="Zip file to extract when --download is set or the data dir is missing.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Mask output directory. Defaults to <data-dir>/output.",
    )
    parser.add_argument(
        "--metadata-path",
        default="preprocessed_data.csv",
        type=Path,
        help="CSV path for image/json/mask metadata.",
    )
    return parser.parse_args()


def download_zip(drive_id: str, zip_path: Path) -> None:
    if shutil.which("gdown") is None:
        raise RuntimeError("gdown is not installed. Install it or download data.zip manually.")

    subprocess.run(
        ["gdown", "--id", drive_id, "-O", str(zip_path)],
        check=True,
    )


def extract_zip(zip_path: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Could not find {zip_path}.")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(zip_path.parent)


def find_annotation_dir(data_dir: Path) -> Path:
    for name in ("mask", "masks"):
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find an annotation directory under {data_dir}.")


def collect_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    return files


def annotation_key(path: Path, root: Path) -> str:
    stem = path.stem
    for suffix in ("_gtFine_polygons", "_gtCoarse_polygons"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return (path.relative_to(root).parent / stem).as_posix()


def image_key(path: Path, root: Path) -> str:
    return (path.relative_to(root).parent / path.stem.split("_")[0]).as_posix()


def pair_images_and_annotations(image_dir: Path, annotation_dir: Path) -> pd.DataFrame:
    image_files = collect_files(image_dir, "*.jpg") + collect_files(image_dir, "*.png")
    annotation_files = collect_files(annotation_dir, "*.json")
    if not image_files:
        raise FileNotFoundError(f"No image files found under {image_dir}.")
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotation files found under {annotation_dir}.")

    annotations_by_key = {
        annotation_key(path, annotation_dir): path for path in annotation_files
    }
    rows = []
    missing = []

    for image_path in sorted(image_files):
        key = image_key(image_path, image_dir)
        annotation_path = annotations_by_key.get(key)
        if annotation_path is None:
            missing.append(str(image_path))
            continue
        rows.append(
            {
                "image": image_path.as_posix(),
                "json": annotation_path.as_posix(),
            }
        )

    if not rows:
        raise RuntimeError("No image/annotation pairs were found.")
    if missing:
        print(f"Warning: skipped {len(missing)} images without matching annotations.")

    return pd.DataFrame(rows)


def read_annotation(path: Path) -> Annotation:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    labels = []
    polygons = []
    for obj in data.get("objects", []):
        label = obj.get("label")
        polygon = obj.get("polygon", [])
        if not label or label not in LABEL_TO_MASK_VALUE or len(polygon) < 3:
            continue
        labels.append(label)
        polygons.append([tuple(vertex) for vertex in polygon])

    return Annotation(
        width=int(data["imgWidth"]),
        height=int(data["imgHeight"]),
        labels=labels,
        polygons=polygons,
    )


def write_mask(annotation_path: Path, annotation_dir: Path, output_dir: Path) -> Path:
    annotation = read_annotation(annotation_path)
    mask = Image.new("L", (annotation.width, annotation.height), color=0)
    draw = ImageDraw.Draw(mask)

    for label, polygon in zip(annotation.labels, annotation.polygons):
        draw.polygon(polygon, fill=LABEL_TO_MASK_VALUE[label])

    relative_path = annotation_path.relative_to(annotation_dir).with_suffix(".png")
    mask_path = output_dir / relative_path
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)
    return mask_path


def preprocess(data_dir: Path, output_dir: Path, metadata_path: Path) -> pd.DataFrame:
    image_dir = data_dir / "images"
    annotation_dir = find_annotation_dir(data_dir)

    dataframe = pair_images_and_annotations(image_dir, annotation_dir)
    mask_paths = []
    for annotation_path in tqdm(dataframe["json"], desc="Generating masks"):
        mask_paths.append(
            write_mask(Path(annotation_path), annotation_dir, output_dir).as_posix()
        )

    dataframe["mask"] = mask_paths
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(metadata_path, index=False)
    return dataframe


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.data_dir / "output"

    try:
        if args.download:
            download_zip(args.drive_id, args.zip_path)
            extract_zip(args.zip_path)
        elif not args.data_dir.exists() and args.zip_path.exists():
            extract_zip(args.zip_path)

        dataframe = preprocess(args.data_dir, output_dir, args.metadata_path)
    except Exception as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(dataframe)} rows to {args.metadata_path}.")
    print(f"Masks are in {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
