

def reg_grid(x,y,z= None, t = None,extend = None):
    if z is None:
        if t is None:
            from .spatial2D_regular_mesh import Grid
            grid = Grid()
            grid.setGrid(x=x,y=y,extend = extend)
            return(grid)
        else:
            from .spatial2D_regular_mesh import Grid
            grid = Grid()
            grid.setGrid(x=x,y=y,extend = extend)
            return(grid)
    else:
        if t is None:
            from .spatial3D_regular_mesh import Grid
            grid = Grid()
            grid.setGrid(x=x,y=y,z = z,extend = extend)
            return(grid)
        else:
            assert False, "4D grids are not supported yet"