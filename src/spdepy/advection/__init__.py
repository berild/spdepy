

def init(grid,var = False):
    if var:
        if grid.sdim == 2:
            from .var_advection2D import VarAdection2D
            return(VarAdection2D(grid))
        else:
            from .var_advection3D import VarAdvection3D
            return(VarAdvection3D(grid))
    else:
        if grid.sdim == 2:
            from .advection2D import Advection2D
            return(Advection2D(grid))
        else:
            from .advection3D import Advection3D
            return(Advection3D(grid))
            
        