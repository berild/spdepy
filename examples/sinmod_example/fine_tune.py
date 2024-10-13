import spdepy as sp
import sys
import numpy as np

def fit(bc):
    data = sp.datasets.get_sinmod_training()
    lr = np.hstack([[0.01]*10,np.linspace(0.1,0.01,200),np.linspace(0.01,0.001,200),np.linspace(0.001,0.0001,200)])  
    
    ### ANI 1.6213
    # p0 = np.load('./fits/var_advection_var_diffusion_ani_bc%d.npy'%bc)
    # mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5),
    #     spde = 'var-whittle-matern',#  parameters = x00,
    #     ha = False, bc = bc, anisotropic = True)
        
    # mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5),
    #     spde = 'var-advection-var-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0)

    # mod.fit(data = data['mut'],verbose = True,lr = lr,
    #         end = "./fits/var_advection_var_diffusion_ani_bc%d"%bc, x0 = p0)

    

    ### ANI bs5 1.6344
    p0 = np.load('./fits/var_advection_var_diffusion_ani_bs5_bc%d.npy'%bc)
    Nbs =  5 
    mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 5, Nbs = Nbs),
        spde = 'var-whittle-matern',#  parameters = x00,
        ha = False, bc = bc, anisotropic = True)
        
    mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 5, Nbs = Nbs),
        spde = 'var-advection-var-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0)

    mod.fit(data = data['mut'],verbose = True,lr = lr,
            end = "./fits/var_advection_var_diffusion_ani_bs5_bc%d"%bc, x0 = p0)


    ### ANI b10 1.6213
    # p0 = np.load('./fits/var_advection_var_diffusion_ani_b10_bc%d.npy'%bc)
    # mod0 = sp.model(grid = sp.grid(x=data['x'], y=data['y'], extend = 10),
    #     spde = 'var-whittle-matern',#  parameters = x00,
    #     ha = False, bc = bc, anisotropic = True)
        
    # mod = sp.model(grid = sp.grid(x=data['x'], y=data['y'], t = data['t'],extend = 10),
    #     spde = 'var-advection-var-diffusion', ha = False, bc = bc, anisotropic = True, mod0 = mod0)

    # mod.fit(data = data['mut'],verbose = True,lr = lr,
    #         end = "./fits/var_advection_var_diffusion_ani_b10_bc%d"%bc, x0 = p0)

    
if __name__ == "__main__":  
    if len(sys.argv) > 1:
        bc = int(sys.argv[1])
    else:
        bc = 3
    fit(bc)
    print("Done")