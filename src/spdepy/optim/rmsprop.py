import numpy as np

class RMSprop:
    def __init__(self,decay = None, memory = None):
        self.decay = 0.5 if decay is None else decay
        self.memory = 0.9 if memory is None else memory
        self.gf2 = None
        self.dx = 0
    
    def __call__(self,x,f,gf,lr)->np.ndarray:
        if self.gf2 is None:
            self.gf2 = gf**2
        else:
            self.gf2 = self.memory*self.gf2 + (1-self.memory) * gf**2
        self.dx = self.decay*self.dx - lr*gf/np.sqrt(self.gf2)
        return(x + self.dx)
    
    def settings(self, **kwargs) -> None:
        if kwargs.get("decay") is not None:
            self.decay = kwargs.get("decay")
        if kwargs.get("memory") is not None:
            self.memory = kwargs.get("memory")