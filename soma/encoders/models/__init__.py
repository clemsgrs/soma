"""Auto-import all encoder model modules to trigger registration."""

from soma.encoders.models import (
    conch,
    gigapath,
    hibou,
    hoptimus,
    midnight,
    phikon,
    uni,
    virchow,
)

__all__: list[str] = []
