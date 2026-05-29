"""Train a lightweight U-Net on preprocessed IDD masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from idd_config import CLASS_VALUES, IMAGE_SIZE, NUM_CLASSES, RANDOM_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net segmentation model.")
    parser.add_argument("--metadata-path", default="preprocessed_data.csv", type=Path)
    parser.add_argument("--model-path", default="models/unet_idd.keras", type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--validation-size", default=0.2, type=float)
    parser.add_argument("--seed", default=RANDOM_STATE, type=int)
    return parser.parse_args()


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training. Install requirements.txt.") from exc
    return tf


class SegmentationSequence:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tf_module,
        batch_size: int,
        shuffle: bool,
        augment: bool,
        seed: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tf = tf_module
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.indexes))
        batch_indexes = self.indexes[start:stop]

        images = []
        masks = []
        for row_index in batch_indexes:
            row = self.dataframe.iloc[row_index]
            image, mask = load_sample(Path(row["image"]), Path(row["mask"]))
            if self.augment:
                image, mask = self.apply_augmentation(image, mask)
            images.append(image)
            masks.append(mask)

        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        return image.copy(), mask.copy()


def load_sample(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    channels = [(mask == value) for value in CLASS_VALUES]
    mask_one_hot = np.stack(channels, axis=-1).astype("float32")
    return image, mask_one_hot


def conv_block(tf, inputs, filters: int):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def build_unet(tf, input_shape=(256, 256, 3), num_classes: int = NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(tf, inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(tf, p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(tf, p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = conv_block(tf, p3, 256)

    u3 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(c4)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c5 = conv_block(tf, u3, 128)
    u2 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(c5)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c6 = conv_block(tf, u2, 64)
    u1 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(c6)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c7 = conv_block(tf, u1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(c7)
    return tf.keras.Model(inputs, outputs, name="idd_unet")


def dice_loss(tf, y_true, y_pred, smooth: float = 1.0):
    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    denominator = tf.reduce_sum(y_true + y_pred, axis=0)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice)


def mean_iou(tf, y_true, y_pred):
    y_true_labels = tf.argmax(y_true, axis=-1)
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    confusion = tf.math.confusion_matrix(
        tf.reshape(y_true_labels, [-1]),
        tf.reshape(y_pred_labels, [-1]),
        num_classes=NUM_CLASSES,
        dtype=tf.float32,
    )
    true_positive = tf.linalg.diag_part(confusion)
    false_positive = tf.reduce_sum(confusion, axis=0) - true_positive
    false_negative = tf.reduce_sum(confusion, axis=1) - true_positive
    denominator = true_positive + false_positive + false_negative
    iou = tf.math.divide_no_nan(true_positive, denominator)
    return tf.reduce_mean(iou)


def train(args: argparse.Namespace) -> None:
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)

    dataframe = pd.read_csv(args.metadata_path)
    missing_columns = {"image", "mask"} - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")

    train_df, val_df = train_test_split(
        dataframe,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )

    sequence_base = tf.keras.utils.Sequence
    train_sequence_cls = type("IDDTrainSequence", (SegmentationSequence, sequence_base), {})
    val_sequence_cls = type("IDDValSequence", (SegmentationSequence, sequence_base), {})
    train_sequence = train_sequence_cls(
        train_df, tf, args.batch_size, shuffle=True, augment=True, seed=args.seed
    )
    val_sequence = val_sequence_cls(
        val_df, tf, args.batch_size, shuffle=False, augment=False, seed=args.seed
    )

    model = build_unet(tf)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=lambda y_true, y_pred: dice_loss(tf, y_true, y_pred),
        metrics=[lambda y_true, y_pred: mean_iou(tf, y_true, y_pred)],
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    print(f"Best model saved to {args.model_path}.")


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
