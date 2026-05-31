"""Platform capabilities and OS detection."""

import os
import sys

IS_POSIX = os.name == "posix"
IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"

# Background jobs rely on POSIX process groups, signals, and fork-like semantics
# which are brittle or behave very differently on Windows.
SUPPORTS_BACKGROUND_JOBS = IS_POSIX
