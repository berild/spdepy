import numpy as np

class Adagrad:
    def __init__(self ) -> None:
        self.g2 = None
        self.dx = 0
    
    def __call__(self, x, f, gf, lr,epsilon=1e-8) -> np.ndarray:
        if self.g2 is None:
            self.g2 = gf**2
        else:
            self.g2 += gf**2
        self.dx = -lr*gf/(np.sqrt(self.g2)+ epsilon)
        return(x + self.dx)
    