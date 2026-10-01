from typing import NamedTuple, Protocol, cast

from torch import Tensor, nn


class BackboneOutput(NamedTuple):
    cls: Tensor
    patch: Tensor


class DinoVisionTransformer(Protocol):
    embed_dim: int
    patch_size: int

    def forward_features(
        self,
        x: Tensor,
        masks: Tensor | None = None,
    ) -> dict[str, Tensor]: ...


class ViTBackbone(nn.Module):
    embed_dim: int
    patch_size: int

    def __init__(
        self,
        model: nn.Module,
        *,
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward_features(self, x: Tensor) -> BackboneOutput:
        out = cast(DinoVisionTransformer, self.model).forward_features(x)
        return BackboneOutput(
            cls=out["x_norm_clstoken"],
            patch=out["x_norm_patchtokens"],
        )
