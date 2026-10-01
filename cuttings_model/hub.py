import torch
from torch import nn

from cuttings_model.backbone import ViTBackbone

DINO_V2_REF = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)


def build(name: str, *, pretrained: bool) -> ViTBackbone:
    obj = torch.hub.load(
        DINO_V2_REF,
        name,
        pretrained=pretrained,
        skip_validation=True,
    )

    if not isinstance(obj, nn.Module):
        raise TypeError(
            f"torch.hub entrypoint {name} returned {type(obj).__name__}"
        )

    embed_dim: int = obj.embed_dim
    patch_size: int = obj.patch_size

    return ViTBackbone(
        obj.eval(),
        embed_dim=embed_dim,
        patch_size=patch_size,
    )
