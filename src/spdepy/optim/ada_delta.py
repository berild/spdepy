import numpy as np

class AdaDelta:
    def __init__(self, rho = None, epsilon = None):
        self.rho = 0.95 if rho is None else rho
        self.epsilon = 1e-8 if epsilon is None else epsilon
        self.egf2 = None
        self.dx = None
        self.edx2 = None
    
    def __call__(self,x,f,gf,lr)->np.ndarray:
        if self.egf2 is None:
            self.egf2 = (1-self.rho)*gf**2
            self.dx = - np.sqrt(self.epsilon)/np.sqrt(self.egf2 + self.epsilon)*gf
            self.edx2 = (1-self.rho)*self.dx**2
        else:
            self.egf2 = self.rho*self.egf2 + (1-self.rho)*gf**2
            self.dx = - np.sqrt(self.edx2 + self.epsilon)/np.sqrt(self.egf2 + self.epsilon)*gf
            self.edx2 = self.rho*self.edx2 + (1-self.rho)*self.dx**2
        return(x + self.dx)
    
    def settings(self, **kwargs) -> None:
        if kwargs.get("rho") is not None:
            self.rho = kwargs.get("rho")
        if kwargs.get("epsilon") is not None:
            self.epsilon = kwargs.get("epsilon")