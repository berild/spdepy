# read version from installed package
from importlib.metadata import version
__version__ = version("spdepy")

#from spdepy.diffusions import diffusion
#from spdepy.advections import advection
from spdepy.spdes import spde_init
from spdepy.grids import grid
from .model import Model
from .optim import Optimize as optim
from .datasets import *

def model(**kwargs) -> Model:
    """model _summary_

   Parameters
    ----------
    grid : Object
        The grid object of the field
    spde: str or int
        Can be either "whittle-matern" or 1, "advection-diffusion" or 2, 
        "advection-var-diffusion" or 3, "cov-advection-diffusion" or 4, 
        "cov-advection-var-diffusion" or 5, "var-advection-diffusion" or 6, 
        or "var-advection-var-diffusion" or 7.
    parameters: np.ndarray, optional
        The parameter vector of all models. Must include the total number of parameters for the model.
    bc: int, optional
        Default 3. Boundary condition. 1: Dirichlet 0, 2: Periodic, 3: Neumann 0
    ha: bool, optional
        Use half angles diffusion, by default True if anisotropic diffusion is used
    anisotropic: bool, optional
        Use anisotropic diffusion, by default True. Unused if ha is True.
    Q0: sparse.csc_matrix, optional
        Default None. The precision matrix of initial field. Required to constructed the precision matrix of spatial temporal field. Can be given later.
    Returns
    -------
    Model: Object
        The model object of the field
    """
    assert kwargs.get("grid") is not None and "grid" in kwargs.get("grid").type , "Grid is not defined"
    ha = True if type(kwargs.get("ha")) is not bool else kwargs.get("ha")
    bc = 3 if type(kwargs.get("bc")) is not int else kwargs.get("bc")
    ani = True if type(kwargs.get("anisotropic")) is not bool else kwargs.get("anisotropic")
    return(Model(spde = kwargs.get("spde"), grid = kwargs.get("grid"), parameters = kwargs.get("parameters"),ani = ani, ha = ha, bc = bc, Q0 = kwargs.get("Q0")))

# Requires Q0 for spatial tmep model