
def diffusion(grid,par = None,bc = 3): #,df = None, var = False, ha = True
    from .anisotropic2D import Anisotropic2D
    return(Anisotropic2D(grid, par = par, bc=bc))
    
    # if df is None:
    #     df = grid.sdim
    # if var:
    #     if  df == 1:
    #         if grid.sdim == 2:
    #             if ha:
    #                 from .var_isotropic2D import VarIsotropic2D
    #             return(VarIsotropic2D(grid))
    #         else:
    #             from .var_isotropic3D import VarIsotropic3D
    #             return(VarIsotropic3D(grid))
    #     elif df == 2:
    #         if grid.sdim == 2:
    #             if not ha:
    #                 from .var_anisotropic2D import VarAnIsotropic2D
    #                 return(VarAnIsotropic2D(grid))
    #             else:
    #                 from .var_ha_anisotropic2D import VarHaAnIsotropic2D
    #                 return(VarHaAnIsotropic2D(grid))
    #         else:
    #             assert False, "Not implemented"
    #     elif df == 3:
    #         if grid.sdim == 3:
    #             if not ha:
    #                 from .var_anisotropic3D import VarAnIsotropic3D
    #                 return(VarAnIsotropic3D(grid))
    #             else:
    #                 assert False, "Not implemented"
    #         else:
    #             assert False, "Not implemented"
    # else:
    #     if  df == 1:
    #         if grid.sdim == 2:
    #             from .isotropic2D import Isotropic2D
    #             return(VarIsotropic2D(grid))
    #         else:
    #             from .isotropic3D import Isotropic3D
    #             return(Isotropic3D(grid))
    #     elif df == 2:
    #         if grid.sdim == 2:
    #             if not ha:
    #                 from .anisotropic2D import AnIsotropic2D
    #                 return(AnIsotropic2D(grid))
    #             else:
    #                 from .ha_anisotropic2D import HaAnIsotropic2D
    #                 return(HaAnIsotropic2D(grid))
    #         else:
    #             assert False, "Not implemented"
    #     elif df == 3:
    #         if grid.sdim == 3:
    #             if not ha:
    #                 from .anisotropic3D import AnIsotropic3D
    #                 return(AnIsotropic3D(grid))
    #             else:
    #                 assert False, "Not implemented"
    #         else:
    #             assert False, "Not implemented"
            
        