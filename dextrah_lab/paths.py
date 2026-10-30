"""Workspace root of the server layout (/data1/users/konstantin.smirnov); DEXTRAH_WORKSPACE points it elsewhere
(e.g. dep-1: ~/robotics/workspace, a directory of symlinks mirroring the server layout)."""
import os
from pathlib import Path

SERVER_WORKSPACE = '/data1/users/konstantin.smirnov'
WORKSPACE = Path(os.environ.get('DEXTRAH_WORKSPACE', SERVER_WORKSPACE))


def remap(path):
    """A server path under SERVER_WORKSPACE, re-rooted at WORKSPACE (other paths unchanged)."""
    s = str(path)
    return Path(str(WORKSPACE)+s[len(SERVER_WORKSPACE):]) if s.startswith(SERVER_WORKSPACE) else Path(s)
