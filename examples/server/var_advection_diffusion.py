import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = sp.datasets.get_sinmod_training()
    lr = np.hstack([[0.01]*10,np.linspace(0.5,0.2,200),np.linspace(0.2,0.1,200)])
    lr2 = np.hstack([[0.01]*10,np.linspace(0.5,0.1,200),np.linspace(0.1,0.01,200)])  
    ### HA
    mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
         spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ha_bc%d.npy'%bc),
         ha = True, bc = bc, anisotropic = True)
    
    mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
         spde = 'var-advection-diffusion',  ha = True, bc = bc, anisotropic = True, mod0 = mod0)
    
    mod.fit(data = data['mut'],verbose = True,lr = lr,
            end = "../fits/var_advection_diffusion_ha_bc%d"%bc)
    
    x0 = mod.getPars()            
    mod.fit(data = data['mut'],verbose = True,lr = lr2, 
            end = "../fits/var_advection_diffusion_ha_bc%d"%bc, x0 = x0)

    ### ANI
    mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
         spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_ani_bc%d.npy'%bc),
         ha = False, bc = bc, anisotropic = True)
    
    mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
         spde = 'var-advection-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0)
    
    mod.fit(data = data['mut'],verbose = True,lr = lr,
            end = "../fits/var_advection_diffusion_ani_bc%d"%bc)
    
    x0 = mod.getPars()            
    mod.fit(data = data['mut'],verbose = True,lr = lr2, 
            end = "../fits/var_advection_diffusion_ani_bc%d"%bc, x0 = x0)
    
    ### ISO
    mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
         spde = 'whittle-matern', parameters = np.load('../fits/whittle_matern_bc%d.npy'%bc),
         ha = False, bc = bc, anisotropic = False)
    
    mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
         spde = 'var-advection-diffusion', ha = False, bc = bc, anisotropic = False, mod0 = mod0)
    
    mod.fit(data = data['mut'],verbose = True,lr = lr,
            end = "../fits/var_advection_diffusion_bc%d"%bc)
    x0 = mod.getPars()            
    mod.fit(data = data['mut'],verbose = True,lr = lr2, 
            end = "../fits/var_advection_diffusion_bc%d"%bc, x0 = x0)
    
if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")