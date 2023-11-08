import numpy as np

class HaAnIsotropic2D:
    
    def H(self,par = None) -> np.ndarray:
        gamma = np.exp(par[0])
        vx = np.exp(par[1])
        vy = np.exp(par[2])
        aV = np.sqrt(vx**2 + vy**2)
        mV = np.array([[vx,vy],[vy,-vx]])
        cosh_aV = (np.exp(aV)+ np.exp(-aV))/2
        sinh_aV = (np.exp(aV)- np.exp(-aV))/2
        H = gamma*cosh_aV*np.eye(2) + gamma*sinh_aV/aV*mV
        return(H)
    
    def dH(self,par,full = True):
        dres = []
        gamma = np.exp(par[0])
        vx = np.exp(par[1])
        vy = np.exp(par[2])
        aV = np.sqrt(vx**2 + vy**2)
        mV = np.array([[vx,vy],[vy,-vx]])
        cosh_aV = (np.exp(aV)+ np.exp(-aV))/2
        sinh_aV = (np.exp(aV)- np.exp(-aV))/2
        # gamma
        dres.append(gamma*cosh_aV*np.eye(2) + gamma*sinh_aV/aV*mV)
        # vx
        dres.append(gamma*vx/aV*(sinh_aV*np.eye(2) + ((cosh_aV - sinh_aV/aV)/aV)*mV) + (gamma*sinh_aV/aV)*np.array([[1,0],[0,-1]]))
        # vy
        dres.append(gamma*vy/aV*(sinh_aV*np.eye(2) + ((cosh_aV - sinh_aV/aV)/aV)*mV) + (gamma*sinh_aV/aV)*np.array([[0,1],[1,0]]))
    
        if full:
            return((dres[0],dres))
        else:
            return(dres)

    def transform(self):
        pass