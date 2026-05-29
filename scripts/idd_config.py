"""Shared configuration for the Indian Driving Dataset segmentation scripts."""

from __future__ import annotations

from collections import OrderedDict


IMAGE_SIZE = (256, 256)
RANDOM_STATE = 42

# IDD labels grouped into the 21 train classes used by the original notebook.
# Values are stored as multiples of 10 to keep generated masks easy to inspect.
LABEL_TO_MASK_VALUE = OrderedDict(
    [
        ("road", 10),
        ("parking", 20),
        ("drivable fallback", 20),
        ("sidewalk", 30),
        ("non-drivable fallback", 40),
        ("rail track", 40),
        ("person", 50),
        ("animal", 50),
        ("rider", 60),
        ("motorcycle", 70),
        ("bicycle", 70),
        ("autorickshaw", 80),
        ("car", 80),
        ("truck", 90),
        ("bus", 90),
        ("vehicle fallback", 90),
        ("trailer", 90),
        ("caravan", 90),
        ("curb", 100),
        ("wall", 100),
        ("fence", 110),
        ("guard rail", 110),
        ("billboard", 120),
        ("traffic sign", 120),
        ("traffic light", 120),
        ("pole", 130),
        ("polegroup", 130),
        ("obs-str-bar-fallback", 130),
        ("building", 140),
        ("bridge", 140),
        ("tunnel", 140),
        ("vegetation", 150),
        ("sky", 160),
        ("fallback background", 160),
        ("unlabeled", 0),
        ("out of roi", 0),
        ("ego vehicle", 170),
        ("ground", 180),
        ("rectification border", 190),
        ("train", 200),
    ]
)

CLASS_VALUES = tuple(sorted(set(LABEL_TO_MASK_VALUE.values())))
NUM_CLASSES = len(CLASS_VALUES)
