"""The managed run: :class:`Pipeline` composes the public verbs in the canonical order.

``run.py`` holds ``Pipeline`` (it extracts through :func:`soma.extract`); ``folds.py``
the canonical per-fold order (``run_fold``); ``build.py`` plan → objects; ``layout.py``
the run layout, ``RunRecorder`` and the resume guards; ``aggregate.py`` the per-fold
writers and the from-disk summary; ``console.py`` the panels; ``mirror.py`` the artifact
mirror; ``representation.py`` the task-free CRoMa evaluation.
"""

from soma.pipeline.folds import FoldResult
from soma.pipeline.run import Pipeline, PipelineResult

__all__ = ["FoldResult", "Pipeline", "PipelineResult"]
