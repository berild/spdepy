import numpy as np

class SGD:
    def __init__(self, decay=None) -> None:
        self.decay = 0.9 if decay is None else decay
        self.dx = 0
    
    def __call__(self,x,f,gf,lr) -> np.ndarray:
        self.dx = self.decay*self.dx - lr*gf 
        return(x + self.dx)
    
    def settings(self, **kwargs) -> None:
        if kwargs.get("decay") is not None:
            self.decay = kwargs.get("decay")
            