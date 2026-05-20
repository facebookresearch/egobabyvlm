# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch a vLLM OpenAI-compatible server (SLURM or local) and connect to it.

vLLM lives in its own pixi environment (``[feature.vllm]`` in ``pixi.toml``)
to keep its torch / numpy / packaging pins from fighting with the main env's
pinned (torch 2.8.0 + xformers + fairseq2) stack. Launch the server from
the ``vllm`` env so the spawned process resolves ``vllm.entrypoints``::

    pixi run -e vllm python -m apps.benchmark_creation.utils.vllm_server \\
        --model google/gemma-4-26B-A4B-it --port 8000

Other code (the OpenAI HTTP client used by ``llm_call`` below, the LT-Swap
generator, etc.) only talks to a *running* vLLM server and lives in the main
env — it doesn't need ``import vllm`` itself.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

import requests
from openai import AsyncOpenAI  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from types import FrameType

logger = logging.getLogger(__name__)

HTTP_OK = 200
SERVER_READY_TIMEOUT_S = 120
JOB_POLL_INTERVAL_S = 5
LOG_TAIL_INTERVAL_S = 3
DEFAULT_REQUEST_TIMEOUT_S = 5
LLM_CALL_TIMEOUT_S = 180.0


def get_api_base(host: str = "localhost", port: int = 8000) -> str:
    """Return the OpenAI-compatible API base URL."""
    return f"http://{host}:{port}/v1"


def get_client(
    host: str = "localhost",
    port: int = 8000,
    api_key: str = "dummy",
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointing at the given host:port."""
    return AsyncOpenAI(api_key=api_key, base_url=get_api_base(host, port))


async def llm_call(  # noqa: PLR0913
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> str | None:
    """Single LLM call with retry + exponential backoff."""
    extra_body: dict[str, object] = {}
    if "qwen" in model.lower():
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(max_retries):
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body if extra_body else None,
                ),
                timeout=LLM_CALL_TIMEOUT_S,
            )
        except (TimeoutError, ConnectionError, OSError) as e:
            wait_time = 2**attempt
            logger.warning(
                "llm_call attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                max_retries,
                e,
                wait_time,
            )
            await asyncio.sleep(wait_time)
        else:
            content = completion.choices[0].message.content
            return content.strip() if content is not None else None
    return None


def wait_for_server(
    host: str = "localhost",
    port: int = 8000,
    timeout: int = 300,
    poll_interval: int = 5,
    proc: subprocess.Popen[bytes] | _LocalProcess | None = None,
) -> bool:
    """Block until the vLLM server responds on ``/v1/models``.

    Args:
        host: Server hostname.
        port: Server port.
        timeout: Total seconds to wait.
        poll_interval: Seconds between probes.
        proc: If provided, the polling loop returns ``False`` immediately if it has exited.

    Returns:
        True if the server is ready, False if the timeout expires or the process exits.
    """
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            logger.error(
                "Server process exited with code %d before becoming ready",
                proc.returncode,
            )
            return False
        try:
            r = requests.get(url, timeout=DEFAULT_REQUEST_TIMEOUT_S)
            if r.status_code == HTTP_OK:
                logger.info("Server ready at %s:%d", host, port)
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(poll_interval)
    logger.warning("Timeout (%ds) waiting for server at %s:%d", timeout, host, port)
    return False


def _build_vllm_cmd(
    model: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the vLLM server command list."""
    served_name = Path(model).name
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        served_name,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


class _LocalProcess:
    """Background vLLM process bundled with its log file handle."""

    def __init__(self, proc: subprocess.Popen[bytes], log_fh: IO[str] | None) -> None:
        self.proc = proc
        self.log_fh = log_fh

    def terminate(self) -> None:
        self.proc.terminate()

    def kill(self) -> None:
        self.proc.kill()

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)

    def poll(self) -> int | None:
        return self.proc.poll()

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode

    @property
    def pid(self) -> int:
        return self.proc.pid

    def close(self) -> None:
        if self.log_fh is not None:
            self.log_fh.close()


def launch_local(  # noqa: PLR0913
    model: str = "google/gemma-4-26B-A4B-it",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    extra_args: list[str] | None = None,
    log_file: str | None = None,
) -> _LocalProcess:
    """Launch a vLLM server as a background subprocess on the current machine.

    Args:
        model: HuggingFace model ID or local path.
        port: Port to serve on.
        max_model_len: Maximum sequence length.
        gpu_memory_utilization: Fraction of GPU memory for KV cache.
        extra_args: Extra CLI flags forwarded to vLLM.
        log_file: Path to redirect stdout/stderr. If None, inherits the parent process streams.

    Returns:
        A wrapper around the server :class:`subprocess.Popen` with optional log file handle.
    """
    cmd = _build_vllm_cmd(model, port, max_model_len, gpu_memory_utilization, extra_args)
    logger.info("Launching vLLM: %s", " ".join(cmd))

    fh: IO[str] | None = None
    if log_file:
        fh = Path(log_file).open("w")  # noqa: SIM115
        try:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        except Exception:
            fh.close()
            raise
    else:
        proc = subprocess.Popen(cmd)

    return _LocalProcess(proc, fh)


def _serve_vllm_slurm(
    model: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    extra_args: list[str],
) -> str:
    """Submitit entry-point: start vLLM and block until the job times out."""
    hostname = socket.gethostname()
    logger.info("[vLLM] Node:  %s", hostname)
    logger.info("[vLLM] Model: %s", model)
    logger.info("[vLLM] Port:  %s", port)
    logger.info("[vLLM] URL:   http://%s:%s/v1", hostname, port)
    logger.info("[vLLM] Max model len: %s", max_model_len)
    logger.info("[vLLM] GPU memory utilization: %s", gpu_memory_utilization)
    sys.stdout.flush()

    conda_prefix = Path(sys.executable).resolve().parents[1]
    conda_lib = conda_prefix / "lib"
    if conda_lib.is_dir():
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}" if ld_path else str(conda_lib)

    cmd = _build_vllm_cmd(model, port, max_model_len, gpu_memory_utilization, extra_args)
    logger.info("[vLLM] Command: %s", " ".join(cmd))
    sys.stdout.flush()

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    def _forward_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info("[vLLM] Received signal %s, shutting down...", signum)
        proc.terminate()

    signal.signal(signal.SIGTERM, _forward_signal)

    proc.wait()
    if proc.returncode != 0:
        msg = f"vLLM server exited with code {proc.returncode}"
        raise RuntimeError(msg)
    return f"vLLM server exited with code {proc.returncode}"


def launch_slurm(  # noqa: PLR0913
    model: str = "google/gemma-4-26B-A4B-it",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    extra_args: list[str] | None = None,
    qos: str | None = None,
    gpus: int = 1,
    cpus: int = 10,
    mem_gb: int = 64,
    timeout_min: int = 120,
) -> None:
    """Submit a vLLM server job to SLURM via submitit.

    ``qos`` is required (cluster-specific; pass via ``--qos`` on the CLI).
    Blocks until the job starts, then prints the connection URL.
    """
    if not qos:
        msg = "qos is required for SLURM submission; pass --qos <your_qos>"
        raise ValueError(msg)

    import submitit

    from apps.benchmark_creation.paths import get_outputs_root

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    checkpoint_root = get_outputs_root()
    log_dir = checkpoint_root / "vLLM_logs" / f"vllm_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model:      %s", model)
    logger.info("Port:       %s", port)
    logger.info("QoS:        %s", qos)
    logger.info("GPUs:       %s", gpus)
    logger.info("Timeout:    %s min", timeout_min)
    logger.info("Log dir:    %s", log_dir)

    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        slurm_qos=qos,
        gpus_per_node=gpus,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=cpus,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
    )

    job = executor.submit(
        _serve_vllm_slurm,
        model,
        port,
        max_model_len,
        gpu_memory_utilization,
        extra_args or [],
    )

    logger.info("Submitted SLURM job: %s", job.job_id)
    logger.info("Logs: %s/%s_0_log.out", log_dir, job.job_id)
    logger.info("Monitor: squeue -j %s", job.job_id)
    logger.info("View logs: tail -f %s/%s_0_log.out", log_dir, job.job_id)

    logger.info("Waiting for job to start...")
    while job.state == "PENDING":
        time.sleep(JOB_POLL_INTERVAL_S)

    if job.state in ("RUNNING", "COMPLETING"):
        log_file = log_dir / f"{job.job_id}_0_log.out"
        node_hostname = _wait_for_node_hostname(job, log_file)
        if node_hostname:
            logger.info("vLLM server starting on: http://%s:%s/v1", node_hostname, port)
        else:
            logger.info("Job is running but couldn't determine hostname from logs.")
            logger.info("Check: tail -f %s", log_file)
    else:
        if job.done():
            try:
                job.result()
            except (RuntimeError, OSError) as e:
                logger.info("Job failed: %s", e)
                return
        logger.info("Job state: %s", job.state)


def _wait_for_node_hostname(job: object, log_file: Path) -> str | None:
    """Tail submitit log to discover the node hostname; return None on timeout/failure."""
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if job.done():  # type: ignore[attr-defined]
            try:
                job.result()  # type: ignore[attr-defined]
            except (RuntimeError, OSError) as e:
                logger.info("ERROR: SLURM job failed: %s", e)
                logger.info("Check logs: tail -f %s", log_file)
                return None
        if log_file.exists():
            text = log_file.read_text()
            for line in text.splitlines():
                if "Node:" in line:
                    return line.split("Node:")[-1].strip()
        time.sleep(LOG_TAIL_INTERVAL_S)
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a vLLM OpenAI-compatible server (SLURM or local).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run vLLM as a local subprocess instead of submitting to SLURM.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-26B-A4B-it",
        help="HuggingFace model ID or local path (default: google/gemma-4-26B-A4B-it).",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server (default: 8000).")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for KV cache (default: 0.90).",
    )
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS (required for SLURM submission).")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument("--mem-gb", type=int, default=64)
    parser.add_argument("--timeout-min", type=int, default=120)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="(--local only) Redirect vLLM output to this file.",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help=(
            "Extra arguments forwarded to vLLM. Use -- separator before flags, "
            "e.g.: python launch_vllm_server.py --model X -- --dtype bfloat16"
        ),
    )
    return parser


def _run_local(args: argparse.Namespace) -> None:
    proc = launch_local(
        model=args.model,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        extra_args=args.extra_args or None,
        log_file=args.log_file,
    )
    try:
        logger.info("vLLM server PID: %s", proc.pid)
        logger.info("Waiting for server at localhost:%s...", args.port)
        if wait_for_server("localhost", args.port, proc=proc.proc):
            logger.info("Ready: http://localhost:%s/v1", args.port)
            logger.info("Press Ctrl+C to stop.")
            try:
                proc.wait()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                proc.wait()
        else:
            logger.info("Server failed to start.")
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            sys.exit(1)
    finally:
        proc.close()


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    if args.local:
        _run_local(args)
    else:
        launch_slurm(
            model=args.model,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_args=args.extra_args or None,
            qos=args.qos,
            gpus=args.gpus,
            cpus=args.cpus,
            mem_gb=args.mem_gb,
            timeout_min=args.timeout_min,
        )


if __name__ == "__main__":
    main()
