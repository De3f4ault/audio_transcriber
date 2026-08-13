import io
import sys

from audiobench.core.settings import get_settings
from audiobench.observatory.types import EventPayload


def format_logfmt(payload: EventPayload) -> str:
    buf = io.StringIO()

    # Fixed-order fields
    for key in ("ts", "level", "subsystem", "event_type"):
        if key in payload:
            buf.write(f"{key}={payload[key]} ")

    if "entity_type" in payload and "entity_id" in payload:
        buf.write(f"entity={payload['entity_type']}:{payload['entity_id']} ")

    for key in ("trace_id", "span_id", "parent_span_id", "duration_ms"):
        if payload.get(key) is not None:
            buf.write(f"{key}={payload[key]} ")

    msg = payload.get("message")
    if msg is not None:
        if " " in msg or "=" in msg:
            buf.write(f'msg="{msg}" ')
        else:
            buf.write(f"msg={msg} ")

    # Remaining
    seen = {"id", "ts", "level", "subsystem", "event_type", "entity_type", "entity_id",
            "trace_id", "span_id", "parent_span_id", "duration_ms", "message"}

    for k, v in payload.items():
        if k not in seen and v is not None:
            if k == "metadata" and isinstance(v, dict):
                for mk, mv in v.items():
                    val_str = str(mv)
                    if " " in val_str or "=" in val_str:
                        buf.write(f'metadata.{mk}="{val_str}" ')
                    else:
                        buf.write(f"metadata.{mk}={val_str} ")
            else:
                val_str = str(v)
                if " " in val_str or "=" in val_str:
                    buf.write(f'{k}="{val_str}" ')
                else:
                    buf.write(f"{k}={val_str} ")

    return buf.getvalue().rstrip()

def write_logfmt_line(payload: EventPayload) -> None:
    settings = get_settings()
    log_path = settings.data_dir / "logs" / "audiobench_obs.log"
    line = format_logfmt(payload)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except OSError as e:
        print(f"[observatory] failed to write logfmt: {e}", file=sys.stderr)
