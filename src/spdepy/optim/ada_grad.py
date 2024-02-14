import numpy as np

class Adagrad:
    def __init__(self , epsilon = 1e-8) -> None:
        self.g2 = None
        self.epsilon = epsilon
    
    def __call__(self, x, f, gf, lr,epsilon=1e-8) -> np.ndarray:
        if self.g2 is None:
            self.g2 = gf**2
        else:
            self.g2 += gf**2
        self.dx = -lr*gf/(np.sqrt(self.g2)+ epsilon)
        return(x + self.dx)
    
    def settings(self, **kwargs) -> None:
        if kwargs.get("epsilon") is not None:
            self.epsilon = kwargs.get("epsilon")
        self.reset()
        pass
    
    def reset(self) -> None:
        self.g2 = None
    
    def printInit(self) -> None:
        print(f"AdaGrad optimizer with epsilon: {self.epsilon}")
        print("--------------------------------------------------------------")