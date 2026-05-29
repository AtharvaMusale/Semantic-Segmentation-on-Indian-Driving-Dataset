# Semantic Segmentation on the Indian Driving Dataset

This project contains the original exploratory notebook plus runnable scripts for
preprocessing IDD polygon annotations, training a U-Net model, and generating
prediction previews.

You can also read the original approach on Medium:
https://atharvamusale.medium.com/semantic-segmentation-on-indian-driving-dataset-3054cb2e70a7

## Project Structure

- `Semantic Segmentation On IDD.ipynb` - original experiment notebook.
- `scripts/idd_config.py` - shared label mapping and image settings.
- `scripts/preprocess_idd.py` - converts polygon JSON annotations to grayscale
  segmentation masks and writes `preprocessed_data.csv`.
- `scripts/train_unet.py` - trains a lightweight TensorFlow/Keras U-Net.
- `scripts/predict_samples.py` - saves side-by-side image, mask, and prediction
  previews.
- `requirements.txt` - Python dependencies for the scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

The scripts expect the extracted dataset under `data/`:

```text
data/
  images/
    <scene>/
      <frame>.jpg
  mask/ or masks/
    <scene>/
      <frame>_gtFine_polygons.json
```

If you already have `data.zip`, place it in the project root. The preprocessing
script will extract it when `data/` is missing.

## Usage

Generate masks and metadata:

```bash
python scripts/preprocess_idd.py
```

Or download with `gdown` first:

```bash
python scripts/preprocess_idd.py --download
```

Train the model:

```bash
python scripts/train_unet.py --epochs 5 --batch-size 8
```

Create prediction previews:

```bash
python scripts/predict_samples.py --limit 20
```

## Notes

The original notebook was written for Google Colab and contains Colab-specific
commands such as `%tensorflow_version`, `!pip install`, and `/content` paths.
The scripts remove those assumptions and use relative project paths instead.
