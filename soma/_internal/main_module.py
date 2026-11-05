"""One name for the code of the launched script, in the launch and in its spawned workers.

``python train.py`` runs the script as the module ``__main__``. A ``spawn`` worker runs it
again as the module ``__mp_main__``, so that its ``if __name__ == "__main__":`` block does
not run again. A class the script defines is then ``__main__.MyEncoder`` in the launch and
``__mp_main__.MyEncoder`` in a worker: the same code under two names.
"""

from __future__ import annotations

__all__ = ["launch_name"]


def launch_name(name: str) -> str:
    """``name``, a module or a dotted ``module.qualname``, as the launch process spells it:
    a leading ``__mp_main__`` becomes ``__main__``."""
    module, dot, rest = name.partition(".")
    return f"__main__{dot}{rest}" if module == "__mp_main__" else name
