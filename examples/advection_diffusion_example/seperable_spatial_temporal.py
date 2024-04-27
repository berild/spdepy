import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = np.load('data/data.npy') 
    grid = np.load('data/grid.npz')
    idx = np.load('data/idxD.npy')
    lr = np.hstack([[0.01]*10,np.linspace(0.5,0.2,200),np.linspace(0.2,0.1,200)])
    lr2 = np.hstack([[0.01]*10,np.linspace(0.3,0.1,200),np.linspace(0.1,0.001,200)]) 
    
    ### ANI
    mod = sp.model(grid = sp.grid(x=grid['x'], y=grid['y'], t = grid['t'],extend = 5),
         spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = True)
    x0 = np.hstack([[-1]*9,[-1]*9,[1]*9,[-1]*9,0.1,0,np.log(1000)])
    mod.fit(data = data,verbose = True,lr = lr,idx = idx, fix = [-1], x0 = x0,
            end = "./fits/seperable_spatial_temporal_ani_bc%d"%bc)
    x0 = mod.getPars(onlySelf=False)            
    mod.fit(data = data,verbose = True,lr = lr2,idx = idx, 
            end = "./fits/seperable_spatial_temporal_ani_bc%d"%bc, x0 = x0)
    
if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")