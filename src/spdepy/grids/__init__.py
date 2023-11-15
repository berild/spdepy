

def grid(x,y,z= None, t = None,extend = None):
    if z is None:
        if t is None:
            from .spatial2D_regular_mesh import Grid
            mesh = Grid()
            mesh.setGrid(x=x,y=y,extend = extend)
            return(mesh)
        else:
            from .spat2Dtemp_regular_mesh import Grid
            mesh = Grid()
            mesh.setGrid(x=x,y=y,extend = extend)
            return(mesh)
    else:
        if t is None:
            from .spatial3D_regular_mesh import Grid
            mesh = Grid()
            mesh.setGrid(x=x,y=y,z = z,extend = extend)
            return(mesh)
        else:
            assert False, "4D grids are not supported yet"