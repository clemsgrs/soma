"""A benchmark's seeds run side by side with ``devices``, one fresh process per seed.

The first seed runs in the calling process and fills the shared feature cache; every
other seed runs in a ``spawn`` worker on a free device. The result must be the one the
sequential loop gives. The specs live in ``tests.benchmark.spawn_benchmarks``: a worker
imports them by name, so it sees no monkeypatch or test-local function.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from soma.benchmark import (
    BenchmarkSpec,
    benchmark_registry,
    register_benchmark,
    run_benchmark,
    run_benchmark_spec,
)
from soma.pipeline.parallel import DeviceJob, run_on_devices
from tests.benchmark import spawn_benchmarks
from tests.e2e.synthetic import make_tile_dataset

CPUS = ["cpu", "cpu"]


@pytest.fixture(scope="module")
def tile_data(tmp_path_factory) -> tuple[Path, Path]:
    return make_tile_dataset(tmp_path_factory.mktemp("tiles"))


@pytest.fixture(scope="module")
def tile_spec_runs(tmp_path_factory, tile_data):
    """The tile spec run sequentially, then with two CPU workers, at the same output root.

    The sequential run's root is moved aside afterwards, so both runs start from an empty
    cache and report the same seed roots.
    """
    dataset_csv, splits_csv = tile_data
    root = tmp_path_factory.mktemp("spec-runs")
    output_root = root / "out"

    def run(devices):
        return run_benchmark_spec(
            spawn_benchmarks.tile_spec(),
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            encoder="fake",
            output_root=output_root,
            devices=devices,
        )

    sequential = run(None)
    output_root.rename(root / "sequential")
    parallel = run(CPUS)
    return sequential, root / "sequential", parallel, output_root


def test_parallel_seeds_give_the_sequential_result_through_run_benchmark_spec(tile_spec_runs):
    sequential, _sequential_root, parallel, output_root = tile_spec_runs

    assert parallel == sequential
    assert parallel.seed_roots == tuple(output_root / f"seed_{seed}" for seed in (0, 1, 2))
    assert [metric.n_seeds for metric in parallel.metrics] == [3, 3]
    assert parallel.metrics[1].std > 0  # the seeds trained different models


def _feature_files(cache_root: Path) -> dict[str, object]:
    """Each cache file's content: the tensors of a ``.pt`` file (``torch.save`` stamps a
    random id into the archive), the bytes of any other."""
    return {
        path.relative_to(cache_root).as_posix(): (
            torch.load(path, map_location="cpu") if path.suffix == ".pt" else path.read_bytes()
        )
        for path in sorted(cache_root.rglob("*"))
        if path.is_file()
    }


def test_a_parallel_run_extracts_once_and_no_worker_touches_the_cache(tile_spec_runs):
    _sequential, sequential_root, _parallel, output_root = tile_spec_runs
    cache_root = output_root / "feature_cache"

    # Each seed's score recorded the cache as it then was; seed 0 runs before any worker.
    snapshots = [
        json.loads((output_root / f"seed_{seed}" / spawn_benchmarks.CACHE_SNAPSHOT).read_text())
        for seed in (0, 1, 2)
    ]
    assert snapshots[0]
    assert snapshots[1] == snapshots[2] == snapshots[0]
    assert spawn_benchmarks.cache_snapshot(cache_root) == snapshots[0]
    assert list(cache_root.rglob("*.tmp")) == []

    sequential_files = _feature_files(sequential_root / "feature_cache")
    parallel_files = _feature_files(cache_root)
    assert sorted(parallel_files) == sorted(sequential_files)
    for name, content in parallel_files.items():
        if isinstance(content, bytes):
            assert content == sequential_files[name], name
        elif torch.is_tensor(content):
            assert torch.equal(content, sequential_files[name]), name
        else:
            assert content.keys() == sequential_files[name].keys(), name


def test_a_spec_that_does_not_pickle_is_refused_before_any_seed_runs(tmp_path, tile_data):
    dataset_csv, splits_csv = tile_data
    spec = BenchmarkSpec(
        name="parallel/lambda",
        canonical_seeds=(0, 1),
        primary_metric=spawn_benchmarks.PRIMARY_METRIC,
        build_config=lambda **kwargs: spawn_benchmarks.build_tile_config(**kwargs),
        score=spawn_benchmarks.score_tile_run,
    )

    with pytest.raises(TypeError, match="module-level functions"):
        run_benchmark_spec(
            spec,
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            encoder="fake",
            output_root=tmp_path / "out",
            devices=CPUS,
        )

    assert not (tmp_path / "out" / "seed_0").exists()


# --- devices --------------------------------------------------------------------------


def _visible(output_root: Path, seeds) -> list[str]:
    return [(output_root / f"seed_{seed}" / "visible").read_text() for seed in seeds]


def test_each_worker_sees_only_its_device_among_the_gpus_visible_to_the_caller(tmp_path, monkeypatch):
    # A SLURM job or a MIG slice already narrows the visible GPUs: cuda:<n> is entry n.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    result = _run_quick(tmp_path, seeds=(0, 1, 2, 3), devices=["cuda:1", 0, "cpu"])

    assert _visible(tmp_path / "out", (1, 2, 3)) == ["5", "3", ""]
    assert result.metrics[0].measured == 1.5  # the mean of the seed numbers 0..3


def test_cuda_n_is_the_nth_gpu_when_the_caller_sees_every_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    _run_quick(tmp_path, seeds=(0, 1, 2), devices=["1", "cuda:0"])

    assert _visible(tmp_path / "out", (1, 2)) == ["1", "0"]


_SCRIPT = """
import os
from pathlib import Path

# Top-level code, which each worker runs again when it imports this script: it records
# the GPUs it may see, as torch would read them if this code touched CUDA.
Path({marks!r}, f"imported-{{os.getpid()}}").write_text(
    os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
)

if __name__ == "__main__":
    from soma.benchmark import run_benchmark_spec
    from tests.benchmark.spawn_benchmarks import quick_spec

    result = run_benchmark_spec(
        quick_spec((0, 1, 2)),
        dataset_csv="dataset.csv",
        splits_csv="splits.csv",
        encoder="fake",
        output_root={output_root!r},
        devices=["cuda:1", "cpu"],
    )
    print("measured", result.metrics[0].measured)
"""


def test_a_script_runs_its_seeds_in_parallel_and_its_top_level_code_already_sees_one_device(
    tmp_path,
):
    # Each worker imports the calling script again before its seed runs. Code at the top
    # of the script (an encoder registration, a torch.cuda check) must already see only
    # the worker's device, and the call under the __main__ guard must not run again.
    marks = tmp_path / "marks"
    marks.mkdir()
    script = tmp_path / "run_seeds.py"
    script.write_text(_SCRIPT.format(marks=str(marks), output_root=str(tmp_path / "out")))
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "3,5", "PYTHONPATH": str(repo_root)}

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "measured 1.0" in completed.stdout
    workers = [(tmp_path / "out" / f"seed_{seed}" / "pid").read_text() for seed in (1, 2)]
    assert sorted((marks / f"imported-{pid}").read_text() for pid in workers) == ["", "5"]
    assert len(list(marks.iterdir())) == 3  # the script itself, then each worker once


@pytest.mark.parametrize(
    ("caller_threads", "devices", "expected"),
    [
        (7, ["cuda:1", "cuda:0", "cpu"], [("5", 2), ("3", 2), ("", 7)]),
        (1, ["cuda:0", "cuda:0"], [("3", 1), ("3", 1)]),
    ],
)
def test_a_gpu_worker_gets_its_share_of_the_callers_cpu_threads_and_a_cpu_worker_all_of_them(
    tmp_path, monkeypatch, caller_threads, devices, expected
):
    # On a GPU, a worker's CPU threads only load data, so the workers split the caller's
    # threads instead of each taking them all. On cpu they do the math: a worker keeps the
    # caller's count, so it computes as the caller would.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    jobs = [
        DeviceJob(f"job {index}", tmp_path / f"job_{index}.log", spawn_benchmarks.thread_count)
        for index in range(len(devices))
    ]
    previous = torch.get_num_threads()
    torch.set_num_threads(caller_threads)
    try:
        results = run_on_devices(jobs, devices)
    finally:
        torch.set_num_threads(previous)

    assert results == expected


def test_a_device_may_take_several_seeds_one_after_another(tmp_path):
    result = _run_quick(tmp_path, seeds=(0, 1, 2), devices=["cpu"])

    assert result.seed_roots == tuple(tmp_path / "out" / f"seed_{seed}" for seed in range(3))
    assert result.metrics[0].measured == 1.0
    assert len(_worker_pids(tmp_path / "out")) == 2  # one fresh process per seed


# --- failures -------------------------------------------------------------------------


def _run_quick(tmp_path, *, seeds=(0, 1, 2), actions=None, devices=CPUS):
    return run_benchmark_spec(
        spawn_benchmarks.quick_spec(seeds, actions),
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        encoder="fake",
        output_root=tmp_path / "out",
        devices=devices,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _worker_pids(output_root: Path) -> list[int]:
    """The processes the quick protocol's seeds ran in, but the calling one."""
    pids = {int(path.read_text()) for path in output_root.glob("seed_*/pid")}
    return sorted(pids - {os.getpid()})


def test_a_seed_that_raises_fails_the_run_with_its_error_naming_the_seed_and_its_log(tmp_path):
    with pytest.raises(ValueError, match="seed 2 failed") as raised:
        _run_quick(tmp_path, actions={2: "raise"})

    log = tmp_path / "out" / "seed_2" / "worker.log"
    assert any("seed 2" in note and str(log) in note for note in raised.value.__notes__)
    assert "ValueError: seed 2 failed" in log.read_text()
    workers = _worker_pids(tmp_path / "out")
    assert len(workers) == 2  # seeds 1 and 2 each ran in a process of their own
    assert not any(_alive(pid) for pid in workers)
    assert multiprocessing.active_children() == []


def test_no_seed_starts_after_a_seed_failed_and_the_running_ones_finish(tmp_path):
    # Seed 1 fails while seed 2 still runs: seed 1's device is free, but seed 3 must not
    # start on it, and seed 2 completes before the error is raised.
    with pytest.raises(ValueError, match="seed 1 failed"):
        _run_quick(tmp_path, seeds=(0, 1, 2, 3), actions={1: "raise", 2: "slow"})

    ran = sorted(path.parent.name for path in (tmp_path / "out").glob("seed_*/pid"))
    assert ran == ["seed_0", "seed_1", "seed_2"]
    assert "seed 2 ran in process" in (tmp_path / "out" / "seed_2" / "worker.log").read_text()


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        ("die", "SIGKILL"),
        ("unpicklable", "could not be sent back"),
        ("unrebuildable", "could not be sent back"),
    ],
)
def test_a_seed_whose_error_cannot_come_back_fails_the_run_with_a_runtime_error(tmp_path, action, reason):
    with pytest.raises(RuntimeError) as raised:
        _run_quick(tmp_path, actions={1: action})

    message = str(raised.value)
    assert message.startswith("seed 1")
    assert reason in message
    assert str(tmp_path / "out" / "seed_1" / "worker.log") in message
    assert not any(_alive(pid) for pid in _worker_pids(tmp_path / "out"))


def test_an_interrupt_stops_every_worker(tmp_path):
    hanging = tmp_path / "out" / "seed_1" / "hanging"
    main_thread = threading.main_thread().ident

    def interrupt_when_seed_1_hangs():
        deadline = time.monotonic() + 60
        while not hanging.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        signal.pthread_kill(main_thread, signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_when_seed_1_hangs, daemon=True)
    interrupter.start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run_quick(tmp_path, actions={1: "hang"})
    interrupter.join()

    assert time.monotonic() - started < 50  # seed 1 sleeps for 60 seconds
    workers = _worker_pids(tmp_path / "out")
    assert workers
    assert not any(_alive(pid) for pid in workers)
    assert multiprocessing.active_children() == []


class _BrokenStartLine:
    """Standard output whose reader has gone away by the time a worker's "started" line comes."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        if "started on" in text:
            raise BrokenPipeError(32, "Broken pipe")
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()


def test_a_worker_is_stopped_when_its_started_line_cannot_be_printed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", _BrokenStartLine(sys.stdout))
    try:
        with pytest.raises(BrokenPipeError):
            _run_quick(tmp_path, actions={1: "hang"})
    finally:
        leaked = multiprocessing.active_children()
        for process in leaked:  # never leave a hanging worker behind a failed check
            process.kill()
            process.join()

    assert leaked == []


@pytest.fixture(scope="module")
def tile_benchmark_runs(tmp_path_factory, tile_data):
    """The registered tile benchmark through ``run_benchmark``, sequential then parallel."""
    dataset_csv, _splits_csv = tile_data
    benchmark = spawn_benchmarks.TileBenchmark()
    register_benchmark(benchmark)
    root = tmp_path_factory.mktemp("benchmark-runs")
    output_root = root / "out"

    def run(devices):
        return run_benchmark(
            benchmark.name,
            encoder="fake",
            curated_dir=dataset_csv.parent,
            output_root=output_root,
            return_result=True,
            devices=devices,
        )

    try:
        sequential = run(None)
        output_root.rename(root / "sequential")
        parallel = run(CPUS)
    finally:
        benchmark_registry._entries.pop(benchmark.name, None)
    return sequential, parallel, output_root


def test_parallel_seeds_give_the_sequential_result_through_run_benchmark(tile_benchmark_runs):
    sequential, parallel, output_root = tile_benchmark_runs

    assert parallel == sequential
    assert parallel.seed_roots == tuple(output_root / f"seed_{seed}" for seed in (0, 1, 2))
    assert [metric.n_seeds for metric in parallel.metrics] == [3, 3]
    assert parallel.metrics[1].std > 0
    # run_benchmark leaves the checksums empty, in parallel as in sequence.
    assert (parallel.dataset_checksum, parallel.splits_checksum, parallel.test_checksum) == ("", "", "")


# --- soma reproduce --devices ---------------------------------------------------------


@pytest.fixture()
def quick_benchmark():
    """Register a quick benchmark built with the given keywords; unregister it after."""
    registered = []

    def register(**kwargs):
        benchmark = spawn_benchmarks.QuickBenchmark(**kwargs)
        register_benchmark(benchmark)
        registered.append(benchmark)
        return benchmark

    yield register
    for benchmark in registered:
        benchmark_registry._entries.pop(benchmark.name, None)


def _reproduce(*argv: str) -> int:
    from soma.cli import main

    try:
        main(["reproduce", *argv])
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_a_missing_reported_metric_in_a_worker_seed_fails_the_cli_as_in_sequence(
    tmp_path, capsys, quick_benchmark
):
    benchmark = quick_benchmark(missing_metric_seeds=(2,))

    def reproduce(name, *extra):
        code = _reproduce(
            benchmark.name, "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / name), *extra
        )
        return code, capsys.readouterr().err

    sequential = reproduce("sequential")
    parallel = reproduce("parallel", "--devices", "cpu,cpu")

    assert sequential == parallel
    assert parallel == (
        2,
        f"Error: benchmark {benchmark.name!r} score is missing Reported metric(s): 'metric/a'.\n",
    )


def test_a_failed_seed_fails_its_panel_cell_and_the_next_encoder_still_runs(
    tmp_path, capsys, monkeypatch, quick_benchmark
):
    import soma.cli as cli

    benchmark = quick_benchmark(failing={"fake": 1})
    monkeypatch.setattr(cli, "_cmd_leaderboard", lambda _args: 0)
    output_root = tmp_path / "out"

    code = _reproduce(
        benchmark.name,
        "--encoders",
        "fake",
        "uni2",
        "--raw-root",
        str(tmp_path / "raw"),
        "--output-root",
        str(output_root),
        "--devices",
        "cpu,cpu",
    )

    assert code == 1
    assert capsys.readouterr().err == (
        "Encoder panel runtime failures (1):\n  - fake: RuntimeError: fake seed 1 failed\n"
    )
    assert sorted(path.parent.name for path in output_root.glob("seed_*/ran-uni2")) == [
        "seed_0",
        "seed_1",
        "seed_2",
    ]


@pytest.mark.parametrize(
    ("devices", "message"),
    [
        ("gpu0", "Error: devices: 'gpu0' is not a device"),
        ("cuda:7", "Error: devices names cuda:7, but this process sees"),
    ],
)
def test_reproduce_refuses_devices_it_cannot_use_before_any_work(
    tmp_path, capsys, monkeypatch, quick_benchmark, devices, message
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    benchmark = quick_benchmark()

    code = _reproduce(
        benchmark.name, "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"), "--devices", devices
    )

    assert code == 2
    assert capsys.readouterr().err.startswith(message)
    assert not (tmp_path / "out").exists()
