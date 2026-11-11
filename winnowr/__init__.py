"""
winnowr: machine-learned genotyping and filtering of germline small-variant calls.

The supported interface is the `winnowr` command line and the files it reads and writes (feature
tables, strlen tables, model files and their `.winnowr.json` sidecar, and the scored VCF). The
Python modules are internal and may change between releases without notice.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("winnowr")
