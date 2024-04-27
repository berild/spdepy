import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = sp.datasets.get_sinmod_training()
    lr = np.hstack([[0.01]*10,np.linspace(0.7,0.5,200),np.linspace(0.5,0.3,200)])
    lr2 = np.hstack([[0.01]*10,np.linspace(0.3,0.1,200),np.linspace(0.1,0.01,200)])  

#     ### Ha
#     mod0 = sp.model(data = sp.data(x=data['x'], y=data['y'], extend = 5),
#          spde = 'whittle-matern', parameters = np.array([-3,-2,-1,-1,-3,3]),
#          ha = True, bc = bc, anisotropic = True)
    
#     mod = sp.model(data = sp.data(x=data['x'], y=data['y'], t = data['t'],extend = 5),
#          spde = 'advection-diffusion',  ha = True, bc = bc, anisotropic = True, Q0 = mod0.mod.Q)
    
#     mod.fit(data = data,verbose = True,lr = lr, 
#             end = "./fits/advection_diffusion_ha_bc%d"%bc)
    
#     x0 = mod.getPars()            
#     mod.fit(data = data,verbose = True,lr = lr2, 
#             end = "./fits/advection_diffusion_ha_bc%d"%bc, x0 = x0)

    ### ANI
    mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
         spde = 'whittle-matern', ha = False, bc = bc, anisotropic = True)
    x00 = np.array([-1,-1,1,1,np.log(100)]) 
    mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
         spde = 'advection-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0)
    x0 = np.hstack([-1,-1,1,-1,1,1,0,x00[:-1],np.log(100)])    
    mod.fit(data = data['mut'],verbose = True,lr = lr,
            end = "./fits/advection_diffusion_ani_bc%d"%bc,fix = [-1], x0 = x0)
    
    x0 = mod.getPars(onlySelf=False)            
    mod.fit(data = data['mut'],verbose = True,lr = lr2,
            end = "./fits/advection_diffusion_ani_bc%d"%bc, x0 = x0)
    
    

if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")