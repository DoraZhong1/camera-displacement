"""Camera Displacement Analyzer.

Measures apparent movement of a physically stationary reference object across
the frames of a recorded video and converts it into estimated camera
displacement (translation in pixels/mm and rotation in degrees).

Public entry points live in :mod:`camera_displacement.analyzer`.
"""

from .analyzer import AnalyzerConfig, DisplacementAnalyzer
from .video_io import CameraRegion, parse_layout

__all__ = ["AnalyzerConfig", "DisplacementAnalyzer", "CameraRegion", "parse_layout"]
__version__ = "1.0.0"
