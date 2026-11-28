"""``predict.split: tiles``: one slide's tiles dealt across one long-lived process per device.

:func:`device_tile_outputs` starts one ``spawn`` process per device, each with every bundle
loaded once on its device (:class:`~soma.evaluation.whole_slide.DeviceTileOutputs`), and
gives a :class:`~soma.evaluation.whole_slide.DealtTileOutputs` over them. The process that
enters it holds no device: it tiles the slides, deals each slide's chunks, takes the
results back in order, stitches them and writes the files.

A device process waits for a request, a slide's own tiles (its chunks, in order), reads
them with ``num_workers`` reader processes, and sends back each batch's raw outputs, one
per bundle, as CPU tensors. It cannot get far ahead: a send blocks until this process reads
the one before, so each device holds about one batch's results. It prints a line per slide
and writes everything else it outputs to ``<log_dir>/worker_<k>.log``.

The devices, their ``CUDA_VISIBLE_DEVICES`` and their CPU threads are those of
:func:`~soma.pipeline.parallel.run_on_devices`. A device process that raises or dies stops
the call: the slide being written is aborted (its writer leaves no partial file), the
other device processes are stopped, and the error names the device and its log. No device
process outlives the call, also on ``KeyboardInterrupt``.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import pickle
import signal
import sys
import traceback
from collections.abc import Iterator, Sequence
from multiprocessing import connection, get_context
from pathlib import Path
from typing import Any

from soma.pipeline.parallel import (
    _EXIT_GRACE_SECONDS,
    _POLL_SECONDS,
    _TERMINATE_GRACE_SECONDS,
    _environment_variable,
    _error_report,
    _job_threads,
    _redirect_output,
    _stop,
    _visible_value,
    resolve_devices,
)

__all__ = ["device_tile_outputs"]


@contextlib.contextmanager
def device_tile_outputs(
    bundles: Sequence[str | Path],
    devices: Sequence[str | int],
    *,
    heads: Sequence[Any],
    num_workers: int,
    log_dir: str | Path,
) -> Iterator[Any]:
    """The bundles at ``bundles`` on each of ``devices``, one process per device, as a
    :class:`~soma.evaluation.whole_slide.DealtTileOutputs`.

    ``heads`` are the bundles' heads, in this process, in bundle order: what the writers
    read (their class scheme, their post-processing). Device ``k`` logs to
    ``<log_dir>/worker_<k>.log``. Every device process has loaded its bundles when this
    enters, and is stopped when it exits.
    """
    import torch

    from soma.evaluation.whole_slide import DealtTileOutputs

    resolved = resolve_devices(list(devices))
    # Read here: a new process starts at torch's default, not at this process's count.
    caller_threads = torch.get_num_threads()
    context = get_context("spawn")
    bundles = [str(path) for path in bundles]
    workers: list[_Worker] = []
    try:
        for k, device in enumerate(resolved):
            threads = _job_threads(device, caller_threads, len(resolved))
            log_path = Path(log_dir) / f"worker_{k}.log"
            workers.append(_Worker.start(context, k, device, threads, log_path, bundles, num_workers))
            print(f"device worker {k} started on {device}; log: {log_path}", flush=True)
        for worker in workers:
            worker.receive("ready")
        yield DealtTileOutputs([_WorkerTileOutputs(worker, list(heads), range(len(bundles))) for worker in workers])
        for worker in workers:
            worker.finish()
    finally:
        for worker in workers:
            worker.stop()


# --- this process's side ---------------------------------------------------------------- #


class _Worker:
    """One device process, and the two pipes to it: requests out, results back."""

    def __init__(self, index: int, device: str, log_path: Path, process, requests, results) -> None:
        self.index, self.device, self.log_path = index, device, log_path
        self.process, self._requests, self._results = process, requests, results
        #: Set while a slide's results are being read: unread results make the pipe unusable.
        self.busy = False

    @classmethod
    def start(cls, context, index, device, threads, log_path: Path, bundles, num_workers) -> "_Worker":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        request_reader, request_writer = context.Pipe(duplex=False)
        result_reader, result_writer = context.Pipe(duplex=False)
        visible = _visible_value(device)
        process = context.Process(
            target=_worker_main,
            args=(request_reader, result_writer, index, device, visible, threads, str(log_path), bundles, num_workers),
            name=f"soma-device-worker-{index}",
            # Not a daemon: its DataLoader starts reader processes of its own.
            daemon=False,
        )
        # The process sees only its device from its first instruction (see parallel._start).
        with _environment_variable("CUDA_VISIBLE_DEVICES", visible):
            process.start()
        # Only the device process holds these ends: its exit then reads as EOF here.
        request_reader.close()
        result_writer.close()
        return cls(index, device, log_path, process, request_writer, result_reader)

    @property
    def label(self) -> str:
        return f"device worker {self.index} on {self.device}"

    def send(self, message: tuple) -> None:
        try:
            self._requests.send_bytes(pickle.dumps(message))
        except OSError:  # a broken pipe: the process has ended
            raise self._ended() from None

    def receive(self, kind: str) -> tuple:
        """The next message from the process, which must be of ``kind``; raises the
        process's error, or one saying it ended without reporting one."""
        while True:
            ready = connection.wait([self._results, self.process.sentinel], timeout=_POLL_SECONDS)
            if self._results in ready:
                try:
                    message = pickle.loads(self._results.recv_bytes())
                except (EOFError, OSError):
                    raise self._ended() from None
            elif self.process.sentinel in ready and not self._results.poll():
                raise self._ended()
            else:
                continue
            if message[0] == kind:
                return message[1:]
            if message[0] in ("error", "unsent"):
                raise self._failed(message)
            raise RuntimeError(f"{self.label} sent {message[0]!r} where {kind!r} was due; its log: {self.log_path}")

    def finish(self) -> None:
        """Ask the process to exit, once every slide is written."""
        with contextlib.suppress(OSError):
            self._requests.send_bytes(pickle.dumps(("stop",)))
        self.process.join(_EXIT_GRACE_SECONDS)

    def stop(self) -> None:
        _stop(self.process)
        self._requests.close()
        self._results.close()

    def _failed(self, message: tuple) -> BaseException:
        kind, payload = message
        where = f"its log: {self.log_path}"
        if kind == "unsent":
            return RuntimeError(f"{self.label} failed with an error that could not be sent back: {payload}. See {where}")
        try:
            error = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001 - the error class does not rebuild here
            return RuntimeError(
                f"{self.label} failed with an error that could not be rebuilt here "
                f"({type(exc).__name__}: {exc}). See {where}"
            )
        error.add_note(f"{self.label} failed; {where}")
        return error

    def _ended(self) -> RuntimeError:
        self.process.join(_TERMINATE_GRACE_SECONDS)
        code = self.process.exitcode
        how = "still running" if code is None else f"exit code {code}"
        if code is not None and code < 0:
            with contextlib.suppress(ValueError):
                how = f"signal {signal.Signals(-code).name}"
        return RuntimeError(
            f"{self.label} ended ({how}) without reporting an error; it may have been killed, "
            f"for example for running out of memory. See its log: {self.log_path}"
        )


class _WorkerTileOutputs:
    """The :class:`~soma.evaluation.whole_slide.TileOutputs` of one device process: the
    bundles ``indices`` of it, with their heads in this process."""

    def __init__(self, worker: _Worker, heads: Sequence[Any], indices: Sequence[int]) -> None:
        self._worker, self._all_heads, self._indices = worker, heads, list(indices)

    @property
    def heads(self) -> list[Any]:
        return [self._all_heads[index] for index in self._indices]

    def subset(self, indices: Sequence[int]) -> "_WorkerTileOutputs":
        return _WorkerTileOutputs(self._worker, self._all_heads, [self._indices[index] for index in indices])

    def batches(self, tiles, *, batch_size: int) -> Iterator[Any]:
        """Send the request now, so that every device starts at once, and read its
        results as they are iterated."""
        worker = self._worker
        if worker.busy:
            raise RuntimeError(f"{worker.label} is still sending another slide's results")
        worker.send(("slide", tiles, int(batch_size), self._indices))
        worker.busy = True
        return self._results(tiles, int(batch_size))

    def _results(self, tiles, batch_size: int) -> Iterator[Any]:
        from soma.evaluation.whole_slide import TileBatch

        worker = self._worker
        for start in range(0, len(tiles.origins), batch_size):
            origins = tiles.origins[start : start + batch_size]
            outputs = _Outputs(worker, len(self._indices), len(origins))
            yield TileBatch(origins, outputs)
            outputs.drain()
        worker.receive("end")
        # Only a slide read to its end leaves the pipe ready for the next request: one left
        # half-read (its writer failed) leaves the process unusable, and the call stops it.
        worker.busy = False


class _Outputs:
    """One batch's outputs, one per bundle, each received when it is read."""

    def __init__(self, worker: _Worker, count: int, num_tiles: int) -> None:
        self._worker, self._left, self._num_tiles = worker, count, num_tiles

    def __iter__(self) -> "_Outputs":
        return self

    def __next__(self):
        from soma.models.model import DenseModelOutput

        if self._left == 0:
            raise StopIteration
        (logits,) = self._worker.receive("output")
        self._left -= 1
        if logits.shape[0] != self._num_tiles:
            raise RuntimeError(
                f"{self._worker.label} sent outputs for {logits.shape[0]} tile(s) where "
                f"{self._num_tiles} were due; its log: {self._worker.log_path}"
            )
        return DenseModelOutput(logits=logits)

    def drain(self) -> None:
        for _ in self:
            pass


# --- the device process ------------------------------------------------------------------ #


def _worker_main(
    requests, results, index: int, device: str, visible: str, threads: int, log_path: str, bundles, num_workers: int
) -> None:
    """A device process: send its output to its log, load the bundles, then predict each
    slide it is sent until it is told to stop (or this process's side is gone)."""
    _redirect_output(log_path)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    print(
        f"--- device worker {index} on {device} (CUDA_VISIBLE_DEVICES={visible!r}, {threads} CPU "
        f"threads), process {os.getpid()}, started {started} ---",
        flush=True,
    )
    try:
        try:
            import torch

            from soma.evaluation.whole_slide import DeviceTileOutputs

            torch.set_num_threads(threads)
            outputs = DeviceTileOutputs.load(bundles, num_workers=num_workers)
            _send(results, ("ready",))
            while True:
                try:
                    message = pickle.loads(requests.recv_bytes())
                except EOFError:
                    break
                if message[0] == "stop":
                    break
                _, tiles, batch_size, indices = message
                count = 0
                for batch in outputs.subset(indices).batches(tiles, batch_size=batch_size):
                    for output in batch.outputs:
                        _send(results, ("output", output.logits.detach().cpu()))
                    count += 1
                _send(results, ("end",))
                print(f"slide {tiles.image.sample_id}: {len(tiles.origins)} tile(s) in {count} batch(es)", flush=True)
        except BaseException as exc:  # noqa: BLE001 - reported to the other side
            traceback.print_exc()
            with contextlib.suppress(OSError):
                _send(results, _error_report(exc))
    finally:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
        results.close()
        requests.close()


def _send(results, message: tuple) -> None:
    # The standard pickle copies a tensor's values: a multiprocessing pickle would share
    # its memory, which this process may free or reuse before the other side reads it.
    results.send_bytes(pickle.dumps(message))
