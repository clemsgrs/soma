"""Liveness tests for the multi-GPU slide-aggregation queue loop."""

import queue

import pytest

torch = pytest.importorskip("torch")

from soma.extraction import slide_aggregation_spawn as sas


class _FakeContext:
    def Queue(self):
        return queue.Queue()


def _run(monkeypatch, fake_process_ctx, preloaded_messages=()):
    q_holder = {}

    class _Ctx(_FakeContext):
        def Queue(self):
            q = queue.Queue()
            for message in preloaded_messages:
                q.put(message)
            q_holder["queue"] = q
            return q

    monkeypatch.setattr(torch.multiprocessing, "get_context", lambda method: _Ctx())
    monkeypatch.setattr(
        torch.multiprocessing,
        "spawn",
        lambda fn, args, nprocs, join: fake_process_ctx,
    )
    shard_completions = []
    written_ids, feature_dim = sas.spawn_slide_aggregation_workers(
        num_workers=2,
        model_name="titan",
        output_variant=None,
        allow_non_recommended_settings=True,
        execution_precision=None,
        execution_batch_size=1,
        execution_num_workers_per_gpu=1,
        execution_prefetch_factor=2,
        output_dir=sas.Path("/tmp"),
        shard_payloads_by_rank=[[{"sample_id": "a"}], [{"sample_id": "b"}]],
        on_shard_complete=lambda ids, dim: shard_completions.append((ids, dim)),
    )
    return written_ids, feature_dim, shard_completions


def test_worker_exception_propagates_instead_of_hanging(monkeypatch):
    class _DeadWorkerCtx:
        def join(self, timeout=None):
            raise RuntimeError("worker died")

    with pytest.raises(RuntimeError, match="worker died"):
        _run(monkeypatch, _DeadWorkerCtx())


def test_workers_exiting_without_results_raises_after_grace(monkeypatch):
    monkeypatch.setattr(sas, "RESULT_DRAIN_GRACE_SECONDS", 0.1)

    class _SilentExitCtx:
        def join(self, timeout=None):
            return True

    with pytest.raises(RuntimeError, match="reported results"):
        _run(monkeypatch, _SilentExitCtx())


def test_happy_path_collects_results(monkeypatch):
    class _HealthyCtx:
        def join(self, timeout=None):
            return True

    messages = [
        {"kind": "progress", "count": 1},
        {"kind": "result", "written_ids": ["a"], "feature_dim": 768},
        {"kind": "result", "written_ids": ["b"], "feature_dim": 768},
    ]
    written_ids, feature_dim, shard_completions = _run(
        monkeypatch, _HealthyCtx(), preloaded_messages=messages
    )
    assert written_ids == {"a", "b"}
    assert feature_dim == 768
    assert [ids for ids, _ in shard_completions] == [["a"], ["b"]]
