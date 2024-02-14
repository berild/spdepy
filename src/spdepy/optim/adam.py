import numpy as np

class Adam:
    def __init__(self, beta1 = None, beta2 = None, epsilon = None):
        self.beta1 = 0.9 if beta1 is None else beta1
        self.beta2 = 0.999 if beta2 is None else beta2
        self.epsilon = 1e-8 if epsilon is None else epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def __call__(self,x,f,gf,lr)->np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = gf
            self.v = gf**2
        else:
            self.m = self.beta1*self.m + (1-self.beta1)*gf
            self.v = self.beta2*self.v + (1-self.beta2)*gf**2
        #print("x = ",np.round(x,3))
        #print("g = ",-np.round(gf, 3))
        mhat = self.m/(1-self.beta1**self.t)
        vhat = self.v/(1-self.beta2**self.t)
        #print("dx= ",- np.round(lr*mhat / (np.sqrt(vhat) + self.epsilon), 3))
        return(x - lr*mhat/(np.sqrt(vhat) + self.epsilon))
    
    
    def settings(self, **kwargs) -> None:
        if kwargs.get("beta1") is not None:
            self.beta1 = kwargs.get("beta1")
        if kwargs.get("beta2") is not None:
            self.beta2 = kwargs.get("beta2")
        if kwargs.get("epsilon") is not None:
            self.epsilon = kwargs.get("epsilon")
        self.reset()
        
    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0
            
    def printInit(self) -> None:
        print(f"Adam optimizer with beta1: {self.beta1}, beta2: {self.beta2}, epsilon: {self.epsilon}")
        print("--------------------------------------------------------------")