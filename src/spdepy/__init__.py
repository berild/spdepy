# read version from installed package
from importlib.metadata import version
__version__ = version("spdepy")

from spdepy.diffusions import diffusion
from spdepy.advections import advection
from .model import Model 
from spdepy.grids import grid

def model(mesh = None, diff = None, adv = None) -> Model:
    mesh = grid() if grid is None else grid
    if isinstance(diff,str):
        diff = diffusion(grid,par = None,bc = 3)
    if isinstance(adv,str):
        adv = advection(grid,par = None,bc = 3)
    return(Model(mesh = mesh, diffusion=diff , advection = adv))