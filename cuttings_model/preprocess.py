import io
import itertools
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rawpy
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from rawpy import (
    ColorSpace,  # ty: ignore[unresolved-import]
    DemosaicAlgorithm,  # ty: ignore[unresolved-import]
    FBDDNoiseReductionMode,  # ty: ignore[unresolved-import]
    HighlightMode,  # ty: ignore[unresolved-import]
)
from torch import Tensor

from cuttings_model.geometry import (
    CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    JPEG_QUALITY,
)


def read_cr2_frame(path: Path) -> npt.NDArray[np.uint8]:
    with rawpy.imread(path.absolute().as_posix()) as raw:
        rgb_image = raw.postprocess(
            demosaic_algorithm=DemosaicAlgorithm.AHD,
            fbdd_noise_reduction=FBDDNoiseReductionMode.Off,
            median_filter_passes=0,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=(2.0, 1.0, 1.3, 0.0),
            output_color=ColorSpace.sRGB,
            output_bps=16,
            no_auto_bright=True,
            auto_bright_thr=0.00,
            bright=1.5,
            exp_shift=None,
            exp_preserve_highlights=0.0,
            no_auto_scale=False,
            user_black=512,
            user_sat=0.0,
            adjust_maximum_thr=0.0,
            gamma=(2.222, 4.5),
            highlight_mode=HighlightMode.Clip,
            user_flip=0,
        )

    clipped = rgb_image[10:-10, 12:-12, :]
    return (clipped >> 8).astype(np.uint8)


def lattice_origins(frame: npt.NDArray[np.uint8]) -> list[tuple[int, int]]:
    height, width = frame.shape[:2]
    rows = range(0, height - CROP_SIZE + 1, CROP_SIZE)
    cols = range(0, width - CROP_SIZE + 1, CROP_SIZE)
    return list(itertools.product(rows, cols))


def _crop_image(
    frame: npt.NDArray[np.uint8],
    origin: tuple[int, int],
) -> Image.Image:
    top, left = origin
    crop = frame[top : top + CROP_SIZE, left : left + CROP_SIZE]
    buffer = io.BytesIO()
    Image.fromarray(crop).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def to_batch(
    frame: npt.NDArray[np.uint8],
    *,
    origins: list[tuple[int, int]],
) -> Tensor:
    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )

    return torch.stack(
        [transform(_crop_image(frame, origin)) for origin in origins]
    )
