import spdepy as sp
import sys
import numpy as np
from tqdm import tqdm

def getTempRes(mod,mu,obs):
    res = np.zeros((obs.shape[0],obs.shape[1],10))
    for i in tqdm(range(obs.shape[1])):
        for j in range(10):
            mod.setModel(mu = mu, sigmas = np.log(np.load("../fits/sinmod/sigmas.npy")),useCov = True)
            uidx = np.array([mod.grid.getIdx([x,y,j],extend = False) for x in range(mod.grid.M) for y in range(mod.grid.N)])
            mod.update(y = obs[uidx,i],idx = uidx)
            res[:,i,j] = mod.grid.getS()@mod.mu
    np.save('../results/temporal/%s.npy'%(mod.spde_type),res)
    
def getSpatRes(mod,mu,obs):
    tmp1 = np.linspace(0,mod.grid.M,4).astype("int32")
    tmp2 = np.linspace(0,mod.grid.N,4).astype("int32")
    res = np.zeros((obs.shape[0],obs.shape[1],9))
    for k in tqdm(range(obs.shape[1])):
        for i in range(3): 
            for j in range(3):
                mod.setModel(mu = mu, sigmas = np.log(np.load("../fits/sinmod/sigmas.npy")),useCov = True)
                uidx = np.array([mod.grid.getIdx([x,y,t],extend = False) for x in np.arange(tmp1[i],tmp1[i+1]) for y in np.arange(tmp2[j],tmp2[j+1]) for t in np.arange(10)])
                mod.update(y = obs[uidx,k],idx = uidx)
                res[:,k,i*3+j] = mod.grid.getS()@mod.mu
    np.save('../results/spatial/%s.npy'%(mod.spde_type),res)
    
def getTempRes2(mod,mu,obs):
    res = np.zeros((obs.shape[0],obs.shape[1],10))
    for i in tqdm(range(obs.shape[1])):
        for j in range(10):
            mod.setModel(mu = mu, sigmas = np.log(np.load("../fits/sinmod/sigmas.npy")),useCov = True)
            uidx = np.array([mod.grid.getIdx([x,y],extend = False) for x in range(mod.grid.M) for y in range(mod.grid.N)])
            mod.update(y = obs[uidx,j,i],idx = uidx)
            res[:,i,j] = mod.grid.getS()@mod.mu
    np.save('../results/temporal/%s.npy'%(mod.spde_type),res)
    
def getSpatRes2(mod,mu,obs):
    tmp1 = np.linspace(0,mod.grid.M,4).astype("int32")
    tmp2 = np.linspace(0,mod.grid.N,4).astype("int32")
    res = np.zeros((obs.shape[0],obs.shape[1],9))
    for k in tqdm(range(obs.shape[1])):
        for i in range(3): 
            for j in range(3):
                mod.setModel(mu = mu, sigmas = np.log(np.load("../fits/sinmod/sigmas.npy")),useCov = True)
                uidx = np.array([mod.grid.getIdx([x,y],extend = False) for x in np.arange(tmp1[i],tmp1[i+1]) for y in np.arange(tmp2[j],tmp2[j+1]) ])
                for t in range(10):
                    mod.update(y = obs[uidx,t,k],idx = uidx)
                res[:,k,i*3+j] = mod.grid.getS()@mod.mu
    np.save('../results/spatial/%s.npy'%(mod.spde_type),res)
    
if __name__ == "__main__":  
    data = sp.datasets.get_sinmod_training()
    test = sp.datasets.get_sinmod_validation()
    if len(sys.argv) == 3: 
        diff = int(sys.argv[1])
        bc = int(sys.argv[2])
        if diff == 1:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                           spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = True)   
            mod.mod.setQ(par = np.load('../fits/seperable_spatial_temporal_ani_bc%d.npy'%bc))
        elif diff == 2:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                           spde = 'seperable-spatial-temporal', ha = True, bc = bc, anisotropic = True)   
            mod.mod.setQ(par = np.load('../fits/seperable_spatial_temporal_ha_bc%d.npy'%bc))
        elif diff == 3:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                           spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = False)   
            mod.mod.setQ(par = np.load('../fits/seperable_spatial_temporal_bc%d.npy'%bc))
        elif diff == 4:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'whittle-matern', ha = False, bc = bc, anisotropic = True)
            mod.mod.setQ(par = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc))
        elif diff == 5:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'whittle-matern', ha = True, bc = bc, anisotropic = True)
            mod.mod.setQ(par = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc))
        elif diff == 6:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'whittle-matern', ha = False, bc = bc, anisotropic = False)
            mod.mod.setQ(par = np.load('../fits/whittle_matern_bc%d.npy'%bc))
        elif diff == 7:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'var-whittle-matern', ha = False, bc = bc, anisotropic = True)
            mod.mod.setQ(par = np.load('../fits/var_whittle_matern_ani_bc%d.npy'%bc))
        elif diff == 8:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'var-whittle-matern', ha = True, bc = bc, anisotropic = True)
            mod.mod.setQ(par = np.load('../fits/var_whittle_matern_ha_bc%d.npy'%bc))
        elif diff == 9:
            mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'],extend = 5),
                 spde = 'var-whittle-matern', ha = False, bc = bc, anisotropic = False)
            mod.mod.setQ(par = np.load('../fits/var_whittle_matern_bc%d.npy'%bc))
        if diff < 4:
            getTempRes(mod, data['muB'], test['data'])
            getSpatRes(mod, data['muB'], test['data'])
        else:
            getTempRes2(mod, data['mu'], test['dataS'])
            getSpatRes2(mod, data['mu'], test['dataS'])
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
                     spde = spde, ha = False, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/advection_diffusion_ani_bc%d.npy'%bc))

            elif adv == 2:
                spde = 'cov-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
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
                     spde = spde, ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                mod.mod.setPars(mod.mod.initFit(data['mut']))
                mod.mod.setQ()
            elif adv == 2:
                spde = 'cov-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                     ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.setPars(mod.mod.initFit(data['mut'],ww = ww))
                mod.mod.setQ()
            elif adv == 3:
                spde = 'var-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                     ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = 'var-advection-diffusion', ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
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
                     spde = spde, ha = False, bc = bc, anisotropic = False, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/advection_diffusion_bc%d.npy'%bc))
            elif adv == 2:
                spde = 'cov-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = False, Q0 = mod0.mod.Q)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.initFit(data['mut'],ww = ww)
                mod.mod.setQ(par = np.load('../fits/cov_advection_diffusion_bc%d.npy'%bc))
            elif adv == 3:
                spde = 'var-advection-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/var_advection_diffusion_bc%d.npy'%bc))
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 4: # Varying anistropic diffusion
            if adv == 1:
                spde = 'advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                     spde = spde, ha = False, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/advection_var_diffusion_ani_bc%d.npy'%bc))
            elif adv == 2:
                spde = 'cov-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.initFit(data['mut'],ww = ww)
                mod.mod.setQ(par = np.load('../fits/cov_advection_var_diffusion_ani_bc%d.npy'%bc))
            elif adv == 3:
                spde = 'var-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                     spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
                     ha = False, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/var_advection_var_diffusion_ani_bc%d.npy'%bc))
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 5: # Varying half-angle anistropic diffusion
            if adv == 1:
                spde = 'advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                        ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/advection_var_diffusion_ha_bc%d.npy'%bc))
            elif adv == 2:
                spde = 'cov-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                        ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.initFit(data['mut'],ww = ww)
                mod.mod.setQ(par = np.load('../fits/cov_advection_var_diffusion_ha_bc%d.npy'%bc))
            elif adv == 3:
                spde = 'var-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
                        ha = True, bc = bc, anisotropic = True)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/var_advection_var_diffusion_ha_bc%d.npy'%bc))
            else:
                assert False, "Advective diffusion not implemented"
        elif diff == 6: # Varying isotropic diffusion
            if adv == 1:
                spde = 'advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                        ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/advection_var_diffusion_bc%d.npy'%bc))
            elif adv == 2:
                spde = 'cov-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                        ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, Q0 = mod0.mod.Q)
                ww = mod.grid.assimilate_adv(data['we'],data['wn'])
                mod.mod.initFit(data['mut'],ww = ww)
                mod.mod.setQ(par = np.load('../fits/cov_advection_var_diffusion_bc%d.npy'%bc))
            elif adv == 3:
                spde = 'var-advection-var-diffusion'
                mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
                        spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
                        ha = False, bc = bc, anisotropic = False)
                mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
                        spde = spde, ha = False, bc = bc, anisotropic = False, Q0 = mod0.mod.Q)
                mod.mod.setQ(par = np.load('../fits/var_advection_var_diffusion_bc%d.npy'%bc))
            else:
                assert False, "Advective diffusion not implemented"
        getTempRes(mod, data['muB'], test['data'])
        getSpatRes(mod, data['muB'], test['data'])
    else:
        assert False, "Grad test not implemented"
    print("Done")