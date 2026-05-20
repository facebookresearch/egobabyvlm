# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base evaluation module utilities."""

import traceback as traceback_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _unwrap_stopes_exception(exc: BaseException) -> BaseException:
    """Unwrap stopes TaskExecutionError to get the inner exception.

    Stopes wraps exceptions in TaskExecutionError, which is a frozen dataclass.
    This can cause issues with Python's exception handling (FrozenInstanceError)
    because Python tries to set __traceback__ on frozen dataclasses.

    Additionally, when a FrozenInstanceError is raised (from tqdm's logging_redirect_tqdm
    context manager), we try to find the real error in the exception chain.

    Args:
        exc: The exception to unwrap.

    Returns:
        The inner exception if exc is a TaskExecutionError, otherwise exc itself.
    """
    if type(exc).__name__ == "FrozenInstanceError":
        if exc.__cause__ is not None:
            return _unwrap_stopes_exception(exc.__cause__)
        if exc.__context__ is not None:
            return _unwrap_stopes_exception(exc.__context__)

    if hasattr(exc, "inner_exception") and exc.inner_exception is not None:
        return _unwrap_stopes_exception(exc.inner_exception)

    return exc


@dataclass
class TaskError:
    """Represents an error from a failed task with full context.

    Unlike storing just {"error": str(e)}, this preserves full traceback information
    and can be serialized to YAML while still allowing proper error propagation.
    """

    task_name: str
    error_type: str
    message: str
    traceback: str

    @classmethod
    def from_exception(cls, task_name: str, exc: BaseException) -> "TaskError":
        """Create a TaskError from an exception.

        Automatically unwraps stopes TaskExecutionError to get the real error.

        Args:
            task_name: Name of the task that failed.
            exc: The exception that was raised.

        Returns:
            A TaskError instance with full error details.
        """
        # Unwrap stopes exceptions to get the real error
        unwrapped = _unwrap_stopes_exception(exc)

        # Try to format traceback, handling frozen dataclass exceptions
        try:
            tb_str = "".join(traceback_module.format_exception(type(unwrapped), unwrapped, unwrapped.__traceback__))
        except Exception:
            # Fallback if traceback formatting fails (e.g., frozen dataclass issues)
            tb_str = f"{type(unwrapped).__name__}: {unwrapped}"

        return cls(
            task_name=task_name,
            error_type=type(unwrapped).__name__,
            message=str(unwrapped),
            traceback=tb_str,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for YAML serialization."""
        return {
            "error": self.message,
            "error_type": self.error_type,
            "traceback": self.traceback,
        }


class PipelineError(Exception):
    """Exception raised when one or more tasks in a pipeline fail.

    This exception aggregates multiple task errors and provides proper
    error propagation through nested pipelines.
    """

    def __init__(
        self,
        pipeline_name: str,
        task_errors: list[TaskError],
        *,
        partial_results: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.task_errors = task_errors
        self.partial_results = partial_results

        # Build error message
        error_msgs = [f"{e.task_name}: {e.error_type}: {e.message}" for e in task_errors]
        message = f"Pipeline '{pipeline_name}' had {len(task_errors)} failed task(s):\n" + "\n".join(
            f"  - {msg}" for msg in error_msgs
        )
        super().__init__(message)

    def get_results_with_errors(self, successful_results: dict[str, Any]) -> dict[str, Any]:
        """Merge successful results with error information.

        Args:
            successful_results: Results from tasks that completed successfully.

        Returns:
            Combined dict with successful results and error dicts for failed tasks.
        """
        # Use partial_results if provided, otherwise use successful_results
        if self.partial_results is not None:
            results = dict(self.partial_results)
        else:
            results = dict(successful_results)
            for task_error in self.task_errors:
                results[task_error.task_name] = task_error.to_dict()
        return results


def process_task_results(
    task_names: list[str],
    task_results: list[Any],
    pipeline_name: str,
    *,
    fail_fast: bool = False,
) -> tuple[dict[str, Any], list[TaskError]]:
    """Process results from asyncio.gather(..., return_exceptions=True).

    Args:
        task_names: Names of the tasks (in same order as results).
        task_results: Results from asyncio.gather with return_exceptions=True.
        pipeline_name: Name of the pipeline (for error messages).
        fail_fast: If True, raise immediately on first error.

    Returns:
        Tuple of (successful_results dict, list of TaskErrors).

    Raises:
        PipelineError: If fail_fast=True and any task failed.
    """
    successful_results: dict[str, Any] = {}
    task_errors: list[TaskError] = []

    for task_name, result in zip(task_names, task_results, strict=True):
        if isinstance(result, BaseException):
            task_error = TaskError.from_exception(task_name, result)
            task_errors.append(task_error)
            if fail_fast:
                raise PipelineError(pipeline_name, task_errors) from result
        else:
            successful_results[task_name] = result

    return successful_results, task_errors


def to_path(path: str) -> Path:
    """Convert a path string to a :class:`pathlib.Path`.

    Args:
        path: A local filesystem path string.

    Returns:
        The path wrapped in a :class:`Path` instance.
    """
    return Path(path)
