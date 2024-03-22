import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = np.load('data/data.npy') 
    grid = np.load('data/grid.npz')
    idx = np.load('data/idxD.npy')
    lr = np.hstack([[0.01]*10,np.linspace(0.5,0.2,200),np.linspace(0.2,0.1,200)])
    lr2 = np.hstack([[0.01]*10,np.linspace(0.5,0.1,200),np.linspace(0.1,0.01,200)])  

    mod0 = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], extend = 5),
         spde = 'whittle-matern', parameters = np.load('data/mod0pars.npy'),
         ha = False, bc = bc, anisotropic = True)
    
    mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], t = grid['t'],extend = 5),
         spde = 'advection-diffusion', ha = False, bc = bc, anisotropic = False, mod0 = mod0)
    
    x0 = np.hstack([-1,-1,1,-1,0,np.load('data/mod0pars.npy')[:-1],np.log(10)])
    mod.fit(data = data,verbose = True,lr = lr,idx = idx, 
            end = "./fits/advection_diffusion_bc%d"%bc, x0 = x0, fix = [-1])
    
    x0 = mod.getPars(onlySelf = False)            
    mod.fit(data = data,verbose = True,lr = lr2,idx = idx, 
            end = "./fits/advection_diffusion_bc%d"%bc, x0 = x0)
    
    

if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")