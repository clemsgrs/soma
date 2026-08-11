"""Reusable encoder metadata for the Croma 0.3 tile-model panel."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from slide2vec.encoders import encoder_registry, resolve_encoder_output


@dataclass(frozen=True, slots=True)
class EncoderOutputSpec:
    """The soma encoder output corresponding to one published model name."""

    soma_encoder: str
    output_variant: str
    dimension: int


CROMA_0_3_ENCODER_PANEL: Mapping[str, EncoderOutputSpec] = MappingProxyType(
    {
        "Virchow2": EncoderOutputSpec("virchow2", "cls_patch_mean", 2560),
        "CONCH": EncoderOutputSpec("conch", "default", 512),
        "GenBio-PathFM": EncoderOutputSpec("genbio-pathfm", "default", 4608),
        "CONCHv1.5": EncoderOutputSpec("conchv15", "default", 768),
        "H0-mini": EncoderOutputSpec("h0-mini", "cls_patch_mean", 1536),
        "Virchow": EncoderOutputSpec("virchow", "cls_patch_mean", 2560),
        "Midnight-12k": EncoderOutputSpec("midnight", "default", 3072),
        "H-optimus-1": EncoderOutputSpec("h-optimus-1", "default", 1536),
        "DINOv2-B": EncoderOutputSpec("dinov2-vitb14", "default", 768),
        "H-optimus-0": EncoderOutputSpec("h-optimus-0", "default", 1536),
        "UNI2-h": EncoderOutputSpec("uni2", "default", 1536),
        "MUSK": EncoderOutputSpec("musk", "ms_aug", 2048),
        "mSTAR": EncoderOutputSpec("mstar", "default", 1024),
        "Prov-GigaPath": EncoderOutputSpec("gigapath", "default", 1536),
        "UNI": EncoderOutputSpec("uni", "default", 1024),
        "Hibou-B": EncoderOutputSpec("hibou-b", "default", 768),
        "GPFM": EncoderOutputSpec("gpfm", "default", 1024),
        "Phikon": EncoderOutputSpec("phikon", "default", 768),
        "Phikon-v2": EncoderOutputSpec("phikonv2", "default", 1024),
        "Prost40M": EncoderOutputSpec("prost40m", "default", 384),
        "Hibou-L": EncoderOutputSpec("hibou-l", "default", 1024),
        "Mascaret": EncoderOutputSpec("mascaret", "default", 1536),
        "Phaet": EncoderOutputSpec("phaet", "default", 1024),
        "RudolfV-2": EncoderOutputSpec("rudolfv2", "cls_patch_mean", 3072),
        "RudolfV-2-B": EncoderOutputSpec("rudolfv2-b", "cls_patch_mean", 1536),
        "RudolfV-2-S": EncoderOutputSpec("rudolfv2-s", "cls_patch_mean", 768),
    }
)


def _metadata_mismatch(
    spec: EncoderOutputSpec,
    metadata: Mapping[str, Any] | None,
    reason: str,
) -> ValueError:
    return ValueError(
        "Croma 0.3 encoder metadata mismatch: "
        f"encoder={spec.soma_encoder!r}, "
        f"requested_variant={spec.output_variant!r}, "
        f"expected_dimension={spec.dimension}, "
        f"observed_metadata={metadata!r}, "
        f"reason={reason}"
    )


def validate_croma_0_3_encoder_panel(
    *, metadata_by_encoder: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    """Validate the pinned panel against slide2vec metadata without loading weights."""
    for spec in CROMA_0_3_ENCODER_PANEL.values():
        try:
            metadata = (
                encoder_registry.info(spec.soma_encoder)
                if metadata_by_encoder is None
                else dict(metadata_by_encoder[spec.soma_encoder])
            )
        except Exception as error:
            raise _metadata_mismatch(spec, None, str(error)) from error
        level = metadata.get("level")
        if level != "tile":
            raise _metadata_mismatch(
                spec,
                metadata,
                f"expected tile-level encoder, observed level={level!r}",
            )
        try:
            resolved = resolve_encoder_output(
                spec.soma_encoder,
                requested_output_variant=spec.output_variant,
                metadata=metadata,
            )
        except Exception as error:
            raise _metadata_mismatch(spec, metadata, str(error)) from error
        if resolved.get("encode_dim") != spec.dimension:
            raise _metadata_mismatch(
                spec,
                metadata,
                f"observed dimension={resolved.get('encode_dim')!r}",
            )
