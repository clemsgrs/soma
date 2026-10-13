"""MIL aggregator implementations."""

# Import to trigger registration
from soma.models.aggregators.mil.abmil import ABMIL
from soma.models.aggregators.mil.clam import CLAM_MB, CLAM_SB
from soma.models.aggregators.mil.dsmil import DSMIL
from soma.models.aggregators.mil.dtfdmil import DTFDMIL
from soma.models.aggregators.mil.hipt import HIPT
from soma.models.aggregators.mil.transmil import TransMIL

__all__ = ["ABMIL", "CLAM_SB", "CLAM_MB", "DSMIL", "DTFDMIL", "HIPT", "TransMIL"]
