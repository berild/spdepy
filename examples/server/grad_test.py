import spdepy as sp
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde

def find_grad(mod, n = 100, h = 0.001):
    pars = mod.getPars(onlySelf = False)
    ngrad = np.zeros(pars.shape[0])
    for i in tqdm(range(pars.shape[0])):
        pars1 = pars.copy()
        pars2 = pars.copy()
        pars1[i] = pars1[i] + h
        pars2[i] = pars2[i] - h
        res1 = mod.mod.logLike(pars1,grad = False)
        res2 = mod.mod.logLike(pars2,grad = False)
        ngrad[i] = (res1 - res2)/(2*h)
        
    sgrad = np.zeros((pars.shape[0],n))
    for i in tqdm(range(n)):
        tmp, tmp2 = mod.mod.logLike(pars,grad = True)
        sgrad[:,i] = tmp2
    
    np.savez('../grad/%s.npz'%(mod.spde_type),ngrad = ngrad, sgrad = sgrad)
    
    


if __name__ == "__main__":  
    data = sp.datasets.get_sinmod_training()
    if len(sys.argv) == 3: 
        diff = int(sys.argv[1])
        bc = int(sys.argv[2])
        if diff == 1:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                           spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = True)   
            mod.mod.setPars(mod.mod.initFit(data['mut']))
            mod.mod.setQ()
        elif diff == 2:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                           spde = 'seperable-spatial-temporal', ha = True, bc = bc, anisotropic = True)   
            mod.mod.setPars(mod.mod.initFit(data['mut']))
            mod.mod.setQ()
        elif diff == 3:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                           spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = False)   
            mod.mod.setPars(mod.mod.initFit(data['mut']))
            mod.mod.setQ()
        elif diff == 4:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'whittle-matern', ha = False, bc = bc, anisotropic = True)
            mod.mod.setPars(mod.mod.initFit(data['muf']))
            mod.mod.setQ()
        elif diff == 5:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'whittle-matern', ha = True, bc = bc, anisotropic = True)
            mod.mod.setPars(mod.mod.initFit(data['muf']))
            mod.mod.setQ()
        elif diff == 6:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'whittle-matern', ha = False, bc = bc, anisotropic = False)
            mod.mod.setPars(mod.mod.initFit(data['muf']))
            mod.mod.setQ()
        elif diff == 7:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'var-whittle-matern', ha = False, bc = bc, anisotropic = True)
            mod.mod.setPars(mod.mod.initFit(data['muf']))
            mod.mod.setQ()
        elif diff == 8:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'var-whittle-matern', ha = True, bc = bc, anisotropic = True)
            mod.mod.setPars(mod.mod.initFit(data['muf']))
            mod.mod.setQ()
        elif diff == 9:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'var-whittle-matern', ha = False, bc = bc, anisotropic = False)
            mod.mod.setPars(mod.mod.initFit(data['muf']))
            mod.mod.setQ()
        find_grad(mod, n = 100)
    elif len(sys.argv) == 4:
        adv = int(sys.argv[1])
        diff = int(sys.argv[2])
        bc = int(sys.argv[3])
        if diff == 1: # Non varying anistropic diffusion
            if adv == 1:
                spde = 'advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = True,mod0 = mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = True,mod0 = mod0)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 2: # Non varying half-angle anistropic diffusion
            if adv == 1:
                spde = 'advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                     ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = True, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                     ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = True, bc = bc, anisotropic = True, mod0=mod0)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                     ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = 'var-advection-diffusion', ha = True, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 3: # Non varying isotropic diffusion
            if adv == 1:
                spde = 'advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = False, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = False, mod0=mod0)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 4: # Varying anistropic diffusion
            if adv == 1:
                spde = 'advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = True, mod0=mod0)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 5: # Varying half-angle anistropic diffusion
            if adv == 1:
                spde = 'advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                        ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = True, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                        ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = True, bc = bc, anisotropic = True, mod0=mod0)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                        ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = True, bc = bc, anisotropic = True, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 6: # Varying isotropic diffusion
            if adv == 1:
                spde = 'advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                        ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                        ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, mod0=mod0)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                        ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, mod0=mod0)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            else:
                assert False, "Advective diffusion not implemented"
        find_grad(mod, n = 100)
    else:
        assert False, "Grad test not implemented"
    print("Done")