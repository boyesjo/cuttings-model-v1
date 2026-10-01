import torch
import torch.nn.functional as F
from torch import Tensor

from cuttings_model.backbone import ViTBackbone
from cuttings_model.geometry import BATCH_SIZE


def crop_features(
    backbone: ViTBackbone,
    batch: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    with torch.inference_mode():
        features = [
            backbone.forward_features(
                batch[start : start + BATCH_SIZE].to(device)
            )
            .cls.float()
            .cpu()
            for start in range(0, batch.shape[0], BATCH_SIZE)
        ]

    return torch.cat(features)


def descriptor(features: Tensor) -> Tensor:
    return F.normalize(features.mean(dim=0), dim=-1)
