"""
sculptor
========

A minimal library for structuring unstructured data with LLMs.
"""

__version__ = "0.1.0"

# Optionally, import top-level classes/functions
from .sculptor import Sculptor
from .sculptor_pipeline import SculptorPipeline

__all__ = [
    "Sculptor",
    "SculptorPipeline"
]