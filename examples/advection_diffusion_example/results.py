import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import spdepy as sp
from tqdm import tqdm
from scipy.stats import norm

def getTempRes(mod):
    mod.setModel()
    idx = np.load('data/idx.npy')
    data = np.load('data/test.npy')
    res = np.zeros((data.shape[1],mod.grid.T))
    res2 = np.zeros((data.shape[1],mod.grid.T))
    n = np.zeros((data.shape[1],mod.grid.T))
    nidx = np.arange(mod.grid.M*mod.grid.N*mod.grid.T)
    for i in tqdm(range(data.shape[1])):
        for t in range(mod.grid.T):
            mod.setModel()
            uidx = np.zeros(mod.grid.M*mod.grid.N*mod.grid.T).astype(bool) 
            tidx = np.array([mod.grid.getIdx([x,y,t],extend = False) for x in range(mod.grid.M) for y in range(mod.grid.N)])
            uidx[tidx] = True
            uidx[idx] = False
            mod.update(y = data[uidx,i] ,idx = nidx[uidx])
            tmp = np.sqrt((mod.grid.getS()@mod.mu - data[:,i])**2).reshape(mod.grid.T, -1)
            mvar = mod.qinv(simple = True)
            z = (data[:,i] - mod.grid.getS()@mod.mu)/np.sqrt(mvar)
            tmp2 = (np.sqrt(mvar) * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - np.sqrt(1/np.pi))).reshape(mod.grid.T, -1)
            res[i,:(mod.grid.T - t)] += [tmp[t+j,~uidx.reshape(mod.grid.T,-1)[t+j,:]].mean() for j in range(mod.grid.T - t)]
            res2[i,:(mod.grid.T - t)] += [tmp2[t+j,~uidx.reshape(mod.grid.T,-1)[t+j,:]].mean() for j in range(mod.grid.T - t)]
            n[i,:(mod.grid.T-t)] += 1
    return (res/n,res2/n)


def findRes(model,bc):
    grid = np.load('data/grid.npz')
    mod0 = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], extend = 5),
            spde = 'whittle-matern', parameters = np.load('data/mod0pars.npy'),
            ha = False, bc = bc, anisotropic = False)
    file = "data/temporal_res4.npz"
    if model == 0:
        parT = np.load('data/modpars.npy')
        mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], t = grid['t'],extend = 5),
                spde = 'var-advection-var-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0, parameters = parT)
        new_resTrue, new_resTrue2 = getTempRes(mod) 
    elif model == 1:
        mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], t = grid['t'],extend = 5), parameters = np.load('./fits/advection_diffusion_ani_bc%d.npy'%bc),
                spde = 'advection-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0)
        new_resAD, new_resAD2 = getTempRes(mod) 
    elif model == 2:
        mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], t = grid['t'],extend = 5), parameters = np.load('./fits/seperable_spatial_temporal_ani_bc%d.npy'%bc),
                spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = True, mod0 = mod0)
        new_resSST, new_resSST2 = getTempRes(mod) 
    elif model == 3:
        mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], t = grid['t'],extend = 5), parameters = np.load('./fits/var_advection_var_diffusion_ani_bc%d.npy'%bc),
                spde = 'var-advection-var-diffusion', ha = False, bc = bc, anisotropic = True, mod0=mod0)
        new_resvAvD, new_resvAvD2 = getTempRes(mod) 
    else:
       raise ValueError("model must be between 0 and 3")
    if os.path.exists(file):
        tmp =  np.load(file,allow_pickle=True)
        resTrue, resTrue2, resAD, resAD2, resSST, resSST2, resvAvD, resvAvD2 = tmp['resTrue'], tmp['resTrue2'], tmp['resAD'], tmp['resAD2'], tmp['resSST'], tmp['resSST2'], tmp['resvAvD'], tmp['resvAvD2'] 
    else:
        resTrue, resTrue2, resAD, resAD2, resSST, resSST2, resvAvD, resvAvD2 = None, None, None, None, None, None, None, None
    if model == 1:
        resAD, resAD2 = new_resAD, new_resAD2
    elif model == 0:
        resTrue, resTrue2 = new_resTrue, new_resTrue2
    elif model == 2:
        resSST, resSST2 = new_resSST, new_resSST2
    elif model == 3:
        resvAvD, resvAvD2 = new_resvAvD, new_resvAvD2
    np.savez(file, resTrue = resTrue, resTrue2 = resTrue2, resAD = resAD, resAD2 = resAD2, resSST = resSST, resSST2 = resSST2, resvAvD = resvAvD, resvAvD2 = resvAvD2)



if __name__ == "__main__":  
    assert len(sys.argv) > 1
    model = int(sys.argv[1])
    bc = int(sys.argv[2])

    findRes(model, bc)
    print("Done")