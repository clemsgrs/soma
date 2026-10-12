"""The managed run: :class:`Pipeline` composes the public verbs in the canonical order.

``run.py`` holds ``Pipeline``; ``folds.py`` the canonical per-fold order
(``run_fold``); ``build.py`` config → objects; ``layout.py`` the run layout,
``RunRecorder`` and the resume guards; ``aggregate.py`` the per-fold writers and the
from-disk summary; ``console.py`` the panels; ``mirror.py`` the artifact mirror;
``representation.py`` the task-free CRoMa evaluation; ``_legacy.py`` the fat-record
bridge.
"""

from soma.pipeline.folds import FoldResult
from soma.pipeline.run import Pipeline, PipelineResult

__all__ = ["FoldResult", "Pipeline", "PipelineResult"]
