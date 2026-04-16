"""
segmentation_gui package — SAGA lightweight segmentation interface.

Sets up sys.path so that scene/, gaussian_renderer/, utils/ etc. are importable
regardless of the working directory when the package is imported.
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .app import main  # noqa: E402

__all__ = ["main"]