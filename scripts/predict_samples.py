"""Create side-by-side prediction previews for a trained IDD segmentation model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idd_config import CLASS_VALUES, IMAGE_SIZE
from train_unet import require_tensorflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render prediction previews.")
    parser.add_argument("--metadata-path", default="preprocessed_data.csv", type=Path)
    parser.add_argument("--model-path", default="models/unet_idd.keras", type=Path)
    parser.add_argument("--output-dir", default="predictions", type=Path)
    parser.add_argument("--limit", default=20, type=int)
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)


def class_index_to_mask(class_indexes: np.ndarray) -> np.ndarray:
    values = np.array(CLASS_VALUES, dtype=np.uint8)
    return values[class_indexes]


def render_predictions(args: argparse.Namespace) -> None:
    tf = require_tensorflow()
    model = tf.keras.models.load_model(args.model_path, compile=False)
    dataframe = pd.read_csv(args.metadata_path).head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in dataframe.iterrows():
        image = load_image(Path(row["image"]))
        mask = load_mask(Path(row["mask"]))
        prediction = model.predict(image[np.newaxis].astype("float32") / 255.0, verbose=0)
        prediction_mask = class_index_to_mask(np.argmax(prediction[0], axis=-1))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis in axes:
            axis.axis("off")
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=max(CLASS_VALUES))
        axes[1].set_title("Mask")
        axes[2].imshow(prediction_mask, cmap="gray", vmin=0, vmax=max(CLASS_VALUES))
        axes[2].set_title("Prediction")

        output_path = args.output_dir / f"sample_{index:03d}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    print(f"Wrote previews to {args.output_dir}.")


def main() -> int:
    args = parse_args()
    try:
        render_predictions(args)
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
