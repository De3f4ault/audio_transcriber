"""AudioBench EventBus — lightweight pub/sub for plugin hooks."""
from audiobench.events.bus import EventBus, get_bus, subscribe

__all__ = ["EventBus", "get_bus", "subscribe"]
