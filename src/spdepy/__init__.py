# read version from installed package
from importlib.metadata import version
__version__ = version("spdepy")

from spdepy.spde import SPDE
from spdepy.model import *
from spdepy.grid import *