import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = sp.datasets.get_sinmod_training()
    lr = np.hstack([[0.01]*10,np.linspace(0.2,0.1,200),np.linspace(0.1,0.1,200)])
    lr2 = np.hstack([[0.01]*10,np.linspace(0.3,0.1,200),np.linspace(0.1,0.001,200)]) 
    ## HA
#     mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
#          spde = "seperable-spatial-temporal", ha = True, bc = bc)
#     mod.fit(data = data['mut'],verbose = True,lr = lr,
#             end = "../fits/seperable_spatial_temporal_ha_bc%d"%bc,fix = [-1])
#     x0 = mod.getPars()            
#     mod.fit(data = data['mut'],verbose = True,lr = lr2, 
#             end = "../fits/seperable_spatial_temporal_ha_bc%d"%bc, x0 = x0,fix = [-1])
    
    ### ANI
    mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
         spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = True)
    x0 = np.hstack([[-1]*9,[-1]*9,[1]*9,[1]*9,0.1,0,np.log(100)])
    mod.fit(data = data['mut'],verbose = True,lr = lr, 
            end = "./fits/seperable_spatial_temporal_ani_bc%d"%bc, x0 = x0, fix = [-1])
    x0 = mod.getPars()
    mod.fit(data = data['mut'],verbose = True,lr = lr2, 
            end = "./fits/seperable_spatial_temporal_ani_bc%d"%bc, x0 = x0)
    
    ### ISO
    # mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
    #      spde = 'seperable-spatial-temporal', ha = False, bc = bc, anisotropic = False)
    # mod.fit(data = data['mut'],verbose = True,lr = lr, 
    #         end = "../fits/seperable_spatial_temporal_bc%d"%bc)
    # x0 = mod.getPars()            
    # mod.fit(data = data['mut'],verbose = True,lr = lr2, 
    #         end = "../fits/seperable_spatial_temporal_bc%d"%bc, x0 = x0)
    
if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")