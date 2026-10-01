from pathlib import Path

import torch
from torch import Tensor

from cuttings_model.backbone import ViTBackbone
from cuttings_model.hub import build

BACKBONE_PREFIXES = ("student_backbone.", "backbone.")


def _under_prefix(
    state_dict: dict[str, Tensor],
    prefix: str,
) -> dict[str, Tensor]:
    return {
        key.removeprefix(prefix): tensor
        for key, tensor in state_dict.items()
        if key.startswith(prefix)
    }


def load_backbone(path: Path) -> ViTBackbone:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hparams = ckpt["hyper_parameters"]
    spec = hparams.get("backbone_spec") or hparams["backbone"]
    weights = next(
        (
            candidate
            for prefix in BACKBONE_PREFIXES
            if (candidate := _under_prefix(ckpt["state_dict"], prefix))
        ),
        ckpt["state_dict"],
    )

    backbone = build(spec["name"], pretrained=False)
    backbone.load_state_dict(weights, strict=True)

    return backbone.eval()
