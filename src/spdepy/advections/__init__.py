

def advection(grid, bc = 3):
    if grid.sdim == 2:
        from .advection2D import Advection2D
        return(Advection2D(grid,bc))
    else:
        assert False, "3D advection is not supported yet"
    # if var:
    #     if grid.sdim == 2:
    #         from .var_advection2D import VarAdection2D
    #         return(VarAdection2D(grid))
    #     else:
    #         assert False, "3D advection is not supported yet"
    # if cov:
    #     if grid.sdim == 2:
    #         from .cov_advection2D import CovAdection2D
    #         return(CovAdection2D(grid))
    #     else:
    #         assert False, "3D advection is not supported yet"
    # else:
    #     if grid.sdim == 2:
    #         from .advection2D import Advection2D
    #         return(Advection2D(grid))
    #     else:
    #         assert False, "3D advection is not supported yet"
            
        