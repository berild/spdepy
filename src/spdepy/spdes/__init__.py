def spde_init(model, grid, parameters = None, ani = True, ha = True, bc = 3, mod0 = None):
    if grid.sdim == 2:
        if (model == "whittle-matern") or model == 1: 
            if ha:
                from .whittle_matern_ha2D import WhittleMaternHa2D
                return(WhittleMaternHa2D(par = parameters, grid = grid, bc = bc)) 
            else:
                if ani:
                    from .whittle_matern_anisotropic2D import WhittleMaternAnisotropic2D
                    return(WhittleMaternAnisotropic2D(par = parameters, grid = grid, bc = bc))
                else:
                    from .whittle_matern2D import WhittleMatern2D
                    return(WhittleMatern2D(par = parameters, grid = grid, bc = bc)) 
        elif (model == "var-whittle-matern") or model == -1:
            if ha:
                from .var_whittle_matern_ha2D import VarWhittleMaternHa2D
                return(VarWhittleMaternHa2D(par = parameters, grid = grid, bc = bc)) 
            else:
                if ani:
                    from .var_whittle_matern_anisotropic2D import VarWhittleMaternAnisotropic2D
                    return(VarWhittleMaternAnisotropic2D(par = parameters, grid = grid, bc = bc))
                else:
                    from .var_whittle_matern2D import VarWhittleMatern2D
                    return(VarWhittleMatern2D(par = parameters, grid = grid, bc = bc)) 
        elif (model == "advection-diffusion") or model == 2:
            if ha:
                from .advection_ha_diffusion2D import AdvectionHaDiffusion2D
                return(AdvectionHaDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .advection_diffusion2D import AdvectionDiffusion2D
                    return(AdvectionDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .advection_idiffusion2D import AdvectionIDiffusion2D
                    return(AdvectionIDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "advection-var-diffusion") or model == 3:
            if ha:
                from .advection_var_ha_diffusion2D import AdvectionVarHaDiffusion2D
                return(AdvectionVarHaDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0)) 
            else:
                if ani:
                    from .advection_var_diffusion2D import AdvectionVarDiffusion2D
                    return(AdvectionVarDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .advection_var_idiffusion2D import AdvectionVarIDiffusion2D
                    return(AdvectionVarIDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "cov-advection-diffusion") or model == 4:
            if ha:
                from .cov_advection_ha_diffusion2D import CovAdvectionHaDiffusion2D
                return(CovAdvectionHaDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .cov_advection_diffusion2D import CovAdvectionDiffusion2D
                    return(CovAdvectionDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .cov_advection_idiffusion2D import CovAdvectionIDiffusion2D
                    return(CovAdvectionIDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "cov-advection-var-diffusion") or model == 5:
            if ha:
                from .cov_advection_var_ha_diffusion2D import CovAdvectionVarHaDiffusion2D
                return(CovAdvectionVarHaDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .cov_advection_var_diffusion2D import CovAdvectionVarDiffusion2D
                    return(CovAdvectionVarDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .cov_advection_var_idiffusion2D import CovAdvectionVarIDiffusion2D 
                    return(CovAdvectionVarIDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "var-advection-diffusion") or model == 6:
            if ha:
                from .var_advection_ha_diffusion2D import VarAdvectionHaDiffusion2D
                return(VarAdvectionHaDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .var_advection_diffusion2D import VarAdvectionDiffusion2D
                    return(VarAdvectionDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .var_advection_idiffusion2D import VarAdvectionIDiffusion2D
                    return(VarAdvectionIDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "var-advection-var-diffusion") or model == 7:
            if ha:
                from .var_advection_var_ha_diffusion2D import VarAdvectionVarHaDiffusion2D
                return(VarAdvectionVarHaDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .var_advection_var_diffusion2D import VarAdvectionVarDiffusion2D
                    return(VarAdvectionVarDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .var_advection_var_idiffusion2D import VarAdvectionVarIDiffusion2D
                    return(VarAdvectionVarIDiffusion2D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "seperable-spatial-temporal") or model == 8:
            if ha:
                from .seperable_spatial_temporal_ha2D import SeperableSpatialTemporalHa2D
                return(SeperableSpatialTemporalHa2D(par = parameters, grid = grid, bc = bc))
            else:
                if ani:
                    from .seperable_spatial_temporal2D import SeperableSpatialTemporal2D
                    return(SeperableSpatialTemporal2D(par = parameters, grid = grid, bc = bc))
                else:
                    from .seperable_spatial_temporal_idiffusion2D import SeperableSpatialTemporalIDiffusion2D
                    return(SeperableSpatialTemporalIDiffusion2D(par = parameters, grid = grid, bc = bc))
        else:
            assert False, "Model not implemented"
    elif grid.sdim == 3:
        assert False, "Model not implemented for 3D grids"
        if (model == "whittle-matern") or model == 1:
            if ha:
                from .whittle_matern_ha3D import WhittleMaternHa3D
                return(WhittleMaternHa3D(par = parameters, grid = grid, bc = bc)) 
            else:
                if ani:
                    from .whittle_matern_anisotropic3D import WhittleMaternAnisotropic3D
                    return(WhittleMaternAnisotropic3D(par = parameters, grid = grid, bc = bc))
                else:
                    from .whittle_matern3D import WhittleMatern3D
                    return(WhittleMatern3D(par = parameters, grid = grid, bc = bc)) 
        elif (model == "advection-diffusion") or model == 2:
            if ha:
                from .advection_ha_diffusion3D import AdvectionHaDiffusion3D
                return(AdvectionHaDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .advection_diffusion3D import AdvectionDiffusion3D
                    return(AdvectionDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .advection_idiffusion3D import AdvectionIDiffusion3D
                    return(AdvectionIDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "advection-var-diffusion") or model == 3:
            if ha:
                from .advection_var_ha_diffusion3D import AdvectionVarHaDiffusion3D
                return(AdvectionVarHaDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0)) 
            else:
                if ani:
                    from .advection_var_diffusion3D import AdvectionVarDiffusion3D
                    return(AdvectionVarDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .advection_var_idiffusion3D import AdvectionVarIDiffusion3D
                    return(AdvectionVarIDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "cov-advection-diffusion") or model == 4:
            if ha:
                from .cov_advection_ha_diffusion3D import CovAdvectionHaDiffusion3D
                return(CovAdvectionHaDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .cov_advection_diffusion3D import CovAdvectionDiffusion3D
                    return(CovAdvectionDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .cov_advection_idiffusion3D import CovAdvectionIDiffusion3D
                    return(CovAdvectionIDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "cov-advection-var-diffusion") or model == 5:
            if ha:
                from .cov_advection_var_ha_diffusion3D import CovAdvectionVarHaDiffusion3D
                return(CovAdvectionVarHaDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .cov_advection_var_diffusion3D import CovAdvectionVarDiffusion3D
                    return(CovAdvectionVarDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .cov_advection_var_idiffusion3D import CovAdvectionVarIDiffusion3D 
                    return(CovAdvectionVarIDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = None))
        elif (model == "var-advection-diffusion") or model == 6:
            if ha:
                from .var_advection_ha_diffusion3D import VarAdvectionHaDiffusion3D
                return(VarAdvectionHaDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .var_advection_diffusion3D import VarAdvectionDiffusion3D
                    return(VarAdvectionDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .var_advection_idiffusion3D import VarAdvectionIDiffusion3DQQ
                    return(VarAdvectionIDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
        elif (model == "var-advection-var-diffusion") or model == 7:
            if ha:
                from .var_advection_var_ha_diffusion3D import VarAdvectionVarHaDiffusion3D
                return(VarAdvectionVarHaDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
            else:
                if ani:
                    from .var_advection_var_diffusion3D import VarAdvectionVarDiffusion3D
                    return(VarAdvectionVarDiffusion3D(par = parameters, grid = grid, bc = bc, mod0 = mod0))
                else:
                    from .var_advection_var_idiffusion3D import VarAdvectionVarIDiffusion3DQQ
                    return(VarAdvectionVarIDiffusion3D(par = parameters, grid = grid, bc = bc, Q0 = Q0))
        else:
            assert False, "Model not implemented"
        