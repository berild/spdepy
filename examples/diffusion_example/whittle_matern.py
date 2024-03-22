import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = np.load('dataS.npy')
    grid = np.load('grid.npz')
    idx = np.load('idxS.npy')
        
    lr = np.hstack([[0.01]*10,np.linspace(0.5,0.2,200),np.linspace(0.2,0.1,200)])
    lr2 = np.hstack([[0.01]*10,np.linspace(0.5,0.1,200),np.linspace(0.1,0.01,200)]) 
    ## HA
    mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'],extend = 5),
         spde = 'whittle-matern', ha = True, bc = bc, anisotropic = True)
    mod.fit(data = data,verbose = True,lr = lr,idx = idx,
            end = "./fits/whittle_matern_ha_bc%d"%bc)
    x0 = mod.getPars()            
    mod.fit(data = data,verbose = True,lr = lr2,idx= idx, 
            end = "./fits/whittle_matern_ha_bc%d"%bc, x0 = x0)
    
    ### ANI
    mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'],extend = 5),
         spde = 'whittle-matern', ha = False, bc = bc, anisotropic = True)
    mod.fit(data = data,verbose = True,lr = lr,idx = idx, 
            end = "./fits/whittle_matern_ani_bc%d"%bc)
    x0 = mod.getPars()            
    mod.fit(data = data,verbose = True,lr = lr2,idx = idx, 
            end = "./fits/whittle_matern_ani_bc%d"%bc, x0 = x0)
    
if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")