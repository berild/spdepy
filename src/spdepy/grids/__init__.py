

def grid(x,y,z= None, t = None,extend = None, Nbs = 3):
    """Constructor for Grid class. 
    Supported domains are Spatial 2D, Spatial 3D or Spatio-temporal 2D are supported.
    These classes also constructs the basis spline functions for the spatially varying parameters within the SPDE.
    See Grid classes for more information.

    Parameters
    ----------
    x : np.ndarray
        x coordinates of grid cell centers in a regular grid.
        Like longitude, or meters in some coordinate system
    y : np.ndarray
        y coordinates of grid cell centers in a regular grid.
        Like latitude, or meters in some coordinate system
    z : np.ndarray, optional
        z coordiantes of grid cell centers in a regular grid of 3D spatial domain
        Like depth or altitude in some coordinate system. If unspecified 2D spatial domain will be used. 
    t : np.ndarrray, optional
        time coordinates. If unspecified, spatial model will be used or error will be raised.
    extend : int, optional
        Optional extension of grid in all direction by extend grid cells. 
        The same step length is used for this extension. If unspecified, no extension is used.
    Returns
    ----------
    Grid
        Grid object of the specified domain.
        Spatial 2D, Spatial 3D or Spatio-temporal 2D are supported.
    """
    if z is None:
        if t is None:
            from .spatial2D_regular_mesh import Grid
            mesh = Grid()
            mesh.setGrid(x=x,y=y,extend = extend, Nbs = Nbs)
            return(mesh)
        else:
            from .spat2Dtemp_regular_mesh import Grid
            mesh = Grid()
            mesh.setGrid(x=x,y=y,t=t,extend = extend, Nbs = Nbs)
            return(mesh)
    else:
        if t is None:
            from .spatial3D_regular_mesh import Grid
            mesh = Grid()
            mesh.setGrid(x=x,y=y,z = z,extend = extend)
            return(mesh)
        else:
            assert False, "4D grids are not supported yet"