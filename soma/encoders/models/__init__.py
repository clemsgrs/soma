"""Auto-import all encoder model modules to trigger registration."""

from soma.encoders.models import (
    conch,
    gigapath,
    hibou,
    hoptimus,
    midnight,
    phikon,
    prism,
    titan,
    uni,
    virchow,
)

__all__: list[str] = []
