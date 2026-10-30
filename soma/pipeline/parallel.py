"""Run independent jobs side by side: one fresh ``spawn`` process per job, one per device.

:func:`run_on_devices` knows nothing about what a job does. It starts each job in a new
process on a free device, at most one job per entry of ``devices`` at a time, hands the
device to the next job when one finishes, sends each job's output to its own log file,
and names the job that failed. The benchmark runners use it to train a benchmark's seeds
in parallel.

A job's process sees one device: ``CUDA_VISIBLE_DEVICES`` is set before the job runs,
so ``cuda:<n>`` (the n-th GPU visible to the caller) is the job's only GPU, and ``cpu``
leaves it none. The job's target and arguments reach the process pickled, so the target
must be a module-level function and its arguments must pickle.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import pickle
import signal
import sys
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing import connection, get_context
from pathlib import Path
from typing import Any

__all__ = ["DeviceJob", "resolve_devices", "run_on_devices"]

#: How often the wait for a job wakes up to notice an interrupt.
_POLL_SECONDS = 1.0
#: How long a job's process may take to exit once it has reported its result.
_EXIT_GRACE_SECONDS = 60.0
#: How long a process may take to stop after ``SIGTERM`` before it is killed.
_TERMINATE_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class DeviceJob:
    """One job for :func:`run_on_devices`.

    ``label`` names the job in messages (``"seed 3"``). ``log_path`` receives everything
    the job's process writes to its standard output and error, including output from C
    code and child processes. ``target`` is called with ``args`` in the job's process;
    its return value, which must pickle, is the job's result.
    """

    label: str
    log_path: Path
    target: Callable[..., Any]
    args: tuple[Any, ...] = ()


def resolve_devices(devices: Sequence[str | int]) -> tuple[str, ...]:
    """Normalise device entries to ``"cpu"`` or ``"cuda:<n>"`` and check each GPU exists.

    An entry is a device string (``"cuda:0"``, ``"cpu"``) or a GPU number (``0`` or
    ``"0"``, meaning ``cuda:0``). ``cuda:<n>`` is the n-th GPU visible to this process:
    entry ``n`` of ``CUDA_VISIBLE_DEVICES`` when it is set. An entry may repeat, to run
    several jobs on one device. Raises ``ValueError`` for an entry that is not a device,
    a GPU this process cannot see, or an empty sequence.
    """
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise ValueError(
            f"devices must be a sequence of device names such as ['cuda:0', 'cuda:1'], "
            f"got {devices!r}"
        )
    if not devices:
        raise ValueError("devices must name at least one device")
    resolved = tuple(_normalise_device(entry) for entry in devices)
    gpus = [int(device.split(":", 1)[1]) for device in resolved if device != "cpu"]
    if gpus:
        visible = _visible_gpu_count()
        missing = sorted({gpu for gpu in gpus if gpu >= visible})
        if missing:
            names = ", ".join(f"cuda:{gpu}" for gpu in missing)
            raise ValueError(
                f"devices names {names}, but this process sees {visible} GPU(s)"
                + _visible_devices_hint()
                + "."
            )
    return resolved


def run_on_devices(
    jobs: Sequence[DeviceJob],
    devices: Sequence[str | int],
    *,
    on_result: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """Run each job in a fresh ``spawn`` process, at most one per device entry at a time.

    Jobs start in order, each on the first free device. When a job finishes, its device
    takes the next job, and ``on_result(index, result)`` runs in this process. Returns the
    results in job order.

    If a job raises, or ``on_result`` raises, no further job starts; the running ones
    finish, then the first error is raised again. A job's own error keeps its type and
    gains a note naming the job and its log. A job whose process dies without reporting
    (killed, for example for running out of memory) or whose error cannot be sent back
    raises ``RuntimeError``. No job process outlives this call, also on
    ``KeyboardInterrupt``.
    """
    resolved = resolve_devices(devices)
    payloads = [pickle.dumps((job.target, job.args)) for job in jobs]
    context = get_context("spawn")
    results: list[Any] = [None] * len(jobs)
    pending = deque(range(len(jobs)))
    free_slots = deque(range(len(resolved)))
    running: dict[int, _Running] = {}
    failure: BaseException | None = None
    try:
        while running or (pending and failure is None):
            while pending and free_slots and failure is None:
                index = pending.popleft()
                slot = free_slots.popleft()
                running[index] = _start(context, jobs[index], payloads[index], resolved[slot], slot)
            # A timeout, so that an interrupt delivered to another thread is still seen.
            ready = connection.wait(
                [entry.reader for entry in running.values()]
                + [entry.process.sentinel for entry in running.values()],
                timeout=_POLL_SECONDS,
            )
            for index, entry in list(running.items()):
                if entry.reader not in ready and entry.process.sentinel not in ready:
                    continue
                message = _finish(entry)
                del running[index]
                free_slots.append(entry.slot)
                job = jobs[index]
                succeeded, value = _outcome(job, entry, message)
                if not succeeded:
                    failure = failure or value
                    continue
                results[index] = value
                print(f"{job.label} finished on {entry.device}; log: {job.log_path}", flush=True)
                if on_result is None:
                    continue
                try:
                    on_result(index, value)
                except KeyboardInterrupt:
                    raise
                except BaseException as exc:  # noqa: BLE001 - re-raised once the others finish
                    failure = failure or exc
    except BaseException:
        for entry in running.values():
            _stop(entry.process)
            entry.reader.close()
        raise
    if failure is not None:
        raise failure
    return results


# --- internals ------------------------------------------------------------------------ #


@dataclass
class _Running:
    process: Any
    reader: connection.Connection
    device: str
    slot: int


def _normalise_device(entry: str | int) -> str:
    if isinstance(entry, int) and not isinstance(entry, bool):
        if entry < 0:
            raise ValueError(f"devices: {entry!r} is not a GPU number")
        return f"cuda:{entry}"
    if isinstance(entry, str):
        text = entry.strip()
        if text == "cpu":
            return "cpu"
        number = text.removeprefix("cuda:")
        if number.isdigit() and number.isascii():
            return f"cuda:{int(number)}"
    raise ValueError(
        f"devices: {entry!r} is not a device; use 'cpu', 'cuda:<n>' or a GPU number <n>"
    )


def _parent_visible_entries() -> list[str] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _visible_gpu_count() -> int:
    entries = _parent_visible_entries()
    if entries is not None:
        return len(entries)
    import torch

    return int(torch.cuda.device_count())


def _visible_devices_hint() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return "" if value is None else f" (CUDA_VISIBLE_DEVICES={value!r})"


def _visible_value(device: str) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` a job on ``device`` gets."""
    if device == "cpu":
        return ""
    gpu = int(device.split(":", 1)[1])
    entries = _parent_visible_entries()
    return str(gpu) if entries is None else entries[gpu]


@contextlib.contextmanager
def _environment_variable(name: str, value: str) -> Iterator[None]:
    """Set ``name`` to ``value`` in this process's environment, then restore it."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _start(context, job: DeviceJob, payload: bytes, device: str, slot: int) -> _Running:
    reader, writer = context.Pipe(duplex=False)
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    visible = _visible_value(device)
    process = context.Process(
        target=_job_main,
        args=(writer, job.label, device, visible, str(log_path), payload),
        name=f"soma-{job.label.replace(' ', '-')}",
        daemon=False,
    )
    # The process inherits this environment, so it sees only its device from its first
    # instruction: before it imports the caller's main module again, whose top-level code
    # may start CUDA, after which CUDA_VISIBLE_DEVICES no longer counts.
    with _environment_variable("CUDA_VISIBLE_DEVICES", visible):
        process.start()
    # Only the job's process may hold the writing end: its exit then reads as EOF here.
    writer.close()
    print(f"{job.label} started on {device}; log: {log_path}", flush=True)
    return _Running(process=process, reader=reader, device=device, slot=slot)


def _finish(entry: _Running) -> tuple | None:
    """Read the job's report, if it sent one, and wait for its process to exit."""
    message: tuple | None = None
    try:
        if entry.reader.poll():
            message = entry.reader.recv()
    except (EOFError, OSError):
        message = None
    finally:
        entry.reader.close()
    entry.process.join(_EXIT_GRACE_SECONDS)
    if entry.process.is_alive():
        _stop(entry.process)
    return message


def _outcome(job: DeviceJob, entry: _Running, message: tuple | None) -> tuple[bool, Any]:
    """``(True, result)`` for a job that succeeded, else ``(False, error to raise)``."""
    where = f"its log: {job.log_path}"
    if message is None:
        code = entry.process.exitcode
        how = f"exit code {code}"
        if code is not None and code < 0:
            try:
                how = f"signal {signal.Signals(-code).name}"
            except ValueError:
                pass
        return False, RuntimeError(
            f"{job.label}: its worker process ended ({how}) without reporting a result; "
            f"it may have been killed, for example for running out of memory. See {where}"
        )
    kind, payload = message
    if kind == "ok":
        return True, payload
    if kind == "unsent":
        return False, RuntimeError(
            f"{job.label} failed in its worker process with an error that could not be "
            f"sent back: {payload}. See {where}"
        )
    try:
        error = pickle.loads(payload)
    except Exception as exc:  # noqa: BLE001 - the error class does not rebuild here
        return False, RuntimeError(
            f"{job.label} failed in its worker process with an error that could not be "
            f"rebuilt here ({type(exc).__name__}: {exc}). See {where}"
        )
    error.add_note(f"{job.label} failed in its worker process; {where}")
    return False, error


def _stop(process) -> None:
    """Stop a job's process: ``SIGTERM``, then ``SIGKILL`` if it lingers."""
    if process.is_alive():
        process.terminate()
        process.join(_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
    process.join()


def _redirect_output(log_path: str) -> None:
    """Send this process's standard output and error to ``log_path``, at the descriptor
    level, so output from C code and from child processes lands there too."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        os.close(descriptor)
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def _job_main(writer, label: str, device: str, visible: str, log_path: str, payload: bytes) -> None:
    """A job's process: send its output to its log, run the job, report.

    ``visible`` is the ``CUDA_VISIBLE_DEVICES`` the process started with (see
    :func:`_start`).
    """
    _redirect_output(log_path)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    print(
        f"--- {label} on {device} (CUDA_VISIBLE_DEVICES={visible!r}), "
        f"process {os.getpid()}, started {started} ---",
        flush=True,
    )
    try:
        try:
            target, args = pickle.loads(payload)
            result = target(*args)
            report = ("ok", result)
            writer.send(report)
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            traceback.print_exc()
            writer.send(_error_report(exc))
    finally:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
        writer.close()


def _error_report(exc: BaseException) -> tuple[str, Any]:
    """The report for a failed job: the pickled error, or its text if it cannot travel."""
    try:
        data = pickle.dumps(exc)
        pickle.loads(data)  # the parent must be able to rebuild it
    except Exception:  # noqa: BLE001 - the error cannot cross the process boundary
        return ("unsent", "".join(traceback.format_exception_only(exc)).strip())
    return ("error", data)
