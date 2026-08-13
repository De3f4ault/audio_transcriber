"""
observatory/process_client.py — @register_with_daemon decorator.

Wraps CLI entry-point callables so the Observatory's Processes panel
can track which subprocesses are currently running.

Design rules:
- Non-fatal: if the daemon is unreachable the wrapped function still runs.
- unregister_process is called in a finally block — exit_code 0 on normal
  return, 1 on exception.
- Uses os.getpid() so each subprocess registers its own PID, not the
  parent's PID.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TypeVar

from audiobench.core.logger_factory import get_logger
from audiobench.daemon.client import DaemonClient

logger = get_logger("observatory.process_client")

F = TypeVar("F", bound=Callable)


def register_with_daemon(*, name: str) -> Callable[[F], F]:
    """Decorator factory. Wraps a callable to register/unregister with daemon.

    Usage::

        @register_with_daemon(name="transcribe")
        def run_transcribe(...):
            ...

    If the daemon is not running, the decorator is a transparent no-op.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            client = DaemonClient()
            pid = os.getpid()

            # Register — non-fatal if daemon unreachable
            try:
                client.register_process(name, pid=pid)
            except Exception as exc:
                logger.debug(
                    "register_with_daemon: could not register '%s' (pid=%d): %s",
                    name, pid, exc,
                )

            exit_code = 0
            try:
                return fn(*args, **kwargs)
            except Exception:
                exit_code = 1
                raise
            finally:
                try:
                    client.unregister_process(name, exit_code=exit_code)
                except Exception as exc:
                    logger.debug(
                        "register_with_daemon: could not unregister '%s': %s",
                        name, exc,
                    )

        return wrapper  # type: ignore[return-value]

    return decorator
