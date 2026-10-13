from __future__ import annotations

import json
import math
import subprocess

import pytest

from examples.beetle import preflight
from examples.beetle.preflight import (
    decoder_batch_attempt,
    probe_decoder_batch_candidates,
    probe_decoder_batch_size,
    validate_decoder_batch_candidates,
)


def test_decoder_probe_runs_the_tracked_training_step_on_cpu() -> None:
    result = probe_decoder_batch_size(batch_size=1, device="cpu", steps=2)

    assert result["passed"] is True
    assert result["batch_size"] == 1
    assert result["steps"] == 2
    assert result["feature_shape"] == [1, 1280, 37, 37]
    assert result["feature_dtype"] == "float32"
    assert result["logits_shape"] == [1, 4, 512, 512]
    assert result["trainable_parameters"] == 1_510_660
    assert result["optimizer"] == {
        "name": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
    }
    assert result["parameters_changed"] is True
    assert math.isfinite(result["final_loss"])
    assert result["samples_per_second"] > 0


def test_decoder_probe_requires_two_complete_optimizer_steps() -> None:
    with pytest.raises(ValueError, match="steps must be an integer >= 2"):
        probe_decoder_batch_size(batch_size=1, device="cpu", steps=1)


@pytest.mark.parametrize(
    "candidates",
    ([], [16, 16, 8], [8, 16], [16, 0, 4], [16, True, 4]),
)
def test_decoder_candidate_list_rejects_invalid_order_or_values(candidates) -> None:
    with pytest.raises(
        ValueError,
        match="positive integers in strictly descending order",
    ):
        validate_decoder_batch_candidates(candidates)


def test_decoder_candidates_run_in_fresh_subprocesses_and_select_largest_passing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    outcomes = {
        64: {"batch_size": 64, "passed": False, "error_type": "OutOfMemoryError"},
        32: {"batch_size": 32, "passed": True, "steps": 2},
        16: {"batch_size": 16, "passed": True, "steps": 2},
    }

    def fake_run(command, **kwargs):
        calls.append(command)
        batch_size = int(command[command.index("--probe-decoder-batch-size") + 1])
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps(outcomes[batch_size]) + "\n",
            stderr="",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    result = probe_decoder_batch_candidates(
        [64, 32, 16], device="cuda:0", steps=2
    )

    assert [row["batch_size"] for row in result["batch_size_attempts"]] == [64, 32, 16]
    assert result["selected_batch_size"] == 32
    assert len(calls) == 3
    assert all(
        command[:3] == [preflight.sys.executable, "-m", "examples.beetle.preflight"]
        for command in calls
    )
    assert all("--decoder-batch-worker" in command for command in calls)


def test_decoder_candidates_record_crashed_worker_and_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="killed")
        batch_size = int(command[command.index("--probe-decoder-batch-size") + 1])
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps({"batch_size": batch_size, "passed": True, "steps": 2}),
            stderr="",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    result = probe_decoder_batch_candidates([8, 4], device="cuda:0")

    assert result["batch_size_attempts"][0] == {
        "batch_size": 8,
        "passed": False,
        "error_type": "SubprocessError",
        "error": "killed",
    }
    assert result["batch_size_attempts"][1]["passed"] is True
    assert result["selected_batch_size"] == 4


def test_decoder_candidates_fail_when_no_candidate_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        batch_size = int(command[command.index("--probe-decoder-batch-size") + 1])
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps(
                {"batch_size": batch_size, "passed": False, "error_type": "OutOfMemoryError"}
            ),
            stderr="",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no decoder batch-size candidate passed"):
        probe_decoder_batch_candidates([8, 4], device="cuda:0")


def test_decoder_batch_attempt_turns_cuda_oom_into_a_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_oom(**kwargs):
        raise pytest.importorskip("torch").OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(preflight, "probe_decoder_batch_size", raise_oom)

    assert decoder_batch_attempt(batch_size=64, device="cuda:0", steps=2) == {
        "batch_size": 64,
        "passed": False,
        "steps": 2,
        "error_type": "OutOfMemoryError",
        "error": "CUDA out of memory",
    }
