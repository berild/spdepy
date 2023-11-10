# read version from installed package
from importlib.metadata import version
__version__ = version("spdepy")

from .diffusion import diffusion
from .advection import advection
from .model import Model 
from .grid import reg_grid

def model(grid = None, diff = None, adv = None) -> Model:
    grid = reg_grid() if grid is None else grid
    if isinstance(diff,str):
        diff = diffusion(grid,par = None,bc = 3)
    if isinstance(adv,str):
        adv = advection(grid,par = None,bc = 3)
    return(Model(grid = grid, diffusion=diff , advection = adv))