"""MIL aggregator implementations."""

# Import to trigger registration
from soma.aggregators.mil.abmil import ABMIL
from soma.aggregators.mil.clam import CLAM
from soma.aggregators.mil.dsmil import DSMIL
from soma.aggregators.mil.dtfdmil import DTFDMIL
from soma.aggregators.mil.hipt import HIPT
from soma.aggregators.mil.transmil import TransMIL

__all__ = ["ABMIL", "CLAM", "DSMIL", "DTFDMIL", "HIPT", "TransMIL"]
