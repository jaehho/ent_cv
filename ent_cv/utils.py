import smtplib
import socket
import sys
import time
import traceback
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, cast

from loguru import logger

from dotenv import load_dotenv, find_dotenv
import os

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

P = ParamSpec("P")
R = TypeVar("R")

GMAIL_ADDRESS  = os.getenv("NOTIFY_EMAIL")
GMAIL_APP_PASS = os.getenv("NOTIFY_APP_PASSWORD")

ResultsFn = Callable[[R], tuple[str, list[Path]]]


def _system_info() -> str:
    """Return a short multi-line string with host and GPU info."""
    lines = [f"Host: {socket.gethostname()}"]
    try:
        import torch
        if torch.cuda.is_available():
            gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            lines.append(f"GPU(s): {', '.join(gpus)}")
        else:
            lines.append("GPU: none (CPU only)")
    except ImportError:
        pass
    return "\n".join(lines)


_MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024  # 4 MB per file (base64 adds ~33% overhead)


def _attach_file(msg: MIMEMultipart, path: Path) -> None:
    """Attach a single file to a MIMEMultipart email message."""
    if not path.exists():
        logger.warning(f"Attachment not found, skipping: {path}")
        return
    size = path.stat().st_size
    if size > _MAX_ATTACHMENT_BYTES:
        logger.warning(
            f"Attachment too large ({size / 1024 / 1024:.1f} MB), skipping: {path}"
        )
        return
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
    msg.attach(part)


def send_email(
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> None:
    """Send a notification email to yourself via Gmail SMTP."""
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        logger.warning(
            "Email notification skipped — NOTIFY_EMAIL or NOTIFY_APP_PASSWORD "
            "not set in .env"
        )
        return

    msg = MIMEMultipart()
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = GMAIL_ADDRESS
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for path in (attachments or []):
        _attach_file(msg, path)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())
        logger.info(f"Notification sent to {GMAIL_ADDRESS}")
    except Exception as e:
        logger.warning(f"Failed to send notification email: {e}")


def notify(
    job_name: str,
    results_fn: ResultsFn | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that sends an email when the wrapped function completes or fails.
    Includes timing, system info, and optional job-specific metrics/attachments.

    If the wrapped function returns None, the notification is suppressed
    (e.g. user aborted before any work was done).

    Args:
        job_name:   Human-readable label shown in the email subject/body.
        results_fn: Optional callable ``(return_value) -> (extra_text, [Path, ...])``.
                    Use this to attach metrics and files that are specific to the job
                    (e.g. YOLO training results, prediction summaries, etc.).

    Usage::

        @notify("Training", results_fn=my_results_fn)
        def main(...):
            ...
            return results_object   # passed to results_fn
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0         = time.monotonic()
            start_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sys_info   = _system_info()
            cmd_str    = " ".join(sys.argv)

            try:
                result = func(*args, **kwargs)

                # None return signals a deliberate early exit (e.g. user aborted) — skip notification
                if result is None:
                    return result

                elapsed      = time.monotonic() - t0
                duration_str = str(timedelta(seconds=int(elapsed)))

                body_parts = [
                    f"{job_name} finished successfully.",
                    "",
                    f"Started:  {start_str}",
                    f"Duration: {duration_str}",
                    "",
                    "── Command ──────────────────────────────────",
                    cmd_str,
                    "",
                    "── System ───────────────────────────────────",
                    sys_info,
                ]

                attachments: list[Path] = []
                if results_fn is not None:
                    extra_text, extra_files = results_fn(result)
                    if extra_text:
                        body_parts += [
                            "",
                            "── Results ──────────────────────────────────",
                            extra_text,
                        ]
                    attachments = [p for p in extra_files if p.exists()]

                send_email(
                    subject=f"✅ {job_name} complete ({duration_str})",
                    body="\n".join(body_parts),
                    attachments=attachments,
                )
                return result

            except Exception:
                elapsed      = time.monotonic() - t0
                duration_str = str(timedelta(seconds=int(elapsed)))

                body_parts = [
                    f"{job_name} failed after {duration_str}.",
                    "",
                    f"Started:  {start_str}",
                    "",
                    "── Command ──────────────────────────────────",
                    cmd_str,
                    "",
                    "── System ───────────────────────────────────",
                    sys_info,
                    "",
                    "── Traceback ────────────────────────────────",
                    traceback.format_exc(),
                ]

                send_email(
                    subject=f"❌ {job_name} failed ({duration_str})",
                    body="\n".join(body_parts),
                )
                raise

        return cast(Callable[P, R], wrapper)
    return decorator