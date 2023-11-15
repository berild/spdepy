import numpy as np
from .rmsprop import RMSprop
from .sgd import SGD
#from .adagrad import Adagrad

class Optimize:
    def __init__(self, fun = None) -> None:
        self.verbose = False
        self.grad = True
        self.histX = []
        self.x = None
        self.histF = []
        self.f = None
        self.lr = 0.1
        self.jac = None
        self.histJac = []
        self.stepType = "rmsprop"
        self.step = RMSprop()
        self.fun = fun
        self.opt_steps = 0
        self.pol = 10
        self.max_steps = 1000
        
    def fit(self, **kwargs) -> np.ndarray:
        """fit _summary_
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial parameters for the model
        kwargs : dict
            Settings for the optimization process. 
            {fun:, stepType:, pol:, truth:, verbose:, lr:, max_steps:}
        """
        assert self.fun is not None or kwargs.get("fun") is not None
        self.settings(**kwargs)
        self.step.settings(**kwargs)
        while not self.stop():
            self.f, self.jac = self.fun(self.x)
            self.fitviz()
            self.histF.append(self.f)
            self.histJac.append(self.jac)
            if hasattr(self.lr, "__len__"):
                lr = self.lr[self.opt_steps]
            self.x = self.step(self.x,self.f,self.jac,lr)
            self.histX.append(self.x)
            self.opt_steps += 1
        self.x = self.polyak()
        res = self.create_res()
        if kwargs.get("end") is not None:
            np.save(kwargs.get("end")+".npy",res['x'])
        return(res)
    
    def fitviz(self) -> None:
        """fitviz prints the current status of the optimization process
        (Maybe be a dashboard later)
        """
        #if self.truth is not None:
        #    print("# %3.0d"%self.opt_steps," fun = %2.4f"%(self.f), "pars = 2.2f"%(self.x - self.truth))
        #else:
        print("# %3.0d"%self.opt_steps,"| fun = %2.4f"%(self.f), self.printPar(self.x))
            
    def settings(self, **kwargs) -> None:
        """settings for the optimization process
        
        Parameters (optional)
        ----------
        x0 : np.ndarray
            Initial parameters for the model
        fun : function
            Function to be optimized
        stepType : str
            Type of step to be used, by default "rmsprop"
        pol : int
            Number of steps to be used for Polyak averaging, by default 10
        truth : np.ndarray
            True parameters, by default None
        verbose : bool
            Verbose output, by default False
        lr : float or np.ndarray
            Learning rate, specified in step class
        max_steps : int
            Maximum number of steps, by default 1000
        """
        self.opt_steps = 0
        if kwargs.get("x0") is not None:
            self.x = kwargs.get("x0")
        if kwargs.get("fun") is not None:
            self.setFun(kwargs.get("fun"))
        if kwargs.get("stepType") is not None:
            self.setStep(kwargs.get("stepType"))
        if kwargs.get("pol") is not None:
            self.pol = kwargs.get("pol")
        if kwargs.get("verbose") is not None:
            self.verbose = kwargs.get("verbose")
            if self.verbose and kwargs.get("print") is not None:
                self.printPar = kwargs.get("print")
        if kwargs.get("lr") is not None:
            self.lr = kwargs.get("lr")
            if hasattr(self.lr, "__len__"):
                self.max_steps = len(self.lr)
        if kwargs.get("max_steps") is not None and self.max_steps is None:
            self.max_steps = kwargs.get("max_steps")
        self.truth = kwargs.get("truth")
            
    def setFun(self,fun) -> None:
        """setFun initializes the function to be optimized
        
        Parameters
        ----------
        fun : function
            Function to be optimized
        """
        self.fun = fun
    
    def setStep(self, stepType: str) -> None: 
        """setStep sets the step type to be used
        
        Parameters
        ----------
        stepType : str
            Type of step to be used, by default "rmsprop"
            options: "rmsprop", "sgd"
        """
        self.stepType = stepType
        if stepType == "rmsprop":
            self.step = RMSprop()
        elif stepType == "sgd":
            self.step = SGD()
        
    def polyak(self) -> np.ndarray:
        """polyak averages the last steps in the optimization process
        
        Returns
        -------
        np.ndarray
            Averaged parameters
        """
        return(np.array(self.histX).T[:,-self.pol:].mean(axis = 1))

    def stop(self):
        """stop checks if the optimization process should be stopped"""
        if self.opt_steps >= self.max_steps:
            return(True)
        return(False)
    
    def create_res(self) -> dict:
        """create_res creates a dictionary with the results from the optimization process

        Returns
        -------
        dict
            Dictionary with the results from the optimization process
        """
        res = {}
        res['method'] = self.stepType
        res['x'] = self.x
        res['fun'] = self.f
        res['jac'] = self.jac
        return(res)
        
    