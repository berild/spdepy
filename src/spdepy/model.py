import numpy as np
from scipy import sparse
from spdepy.spdes import spde_init
from spdepy.optim import Optimize


class Model:
    '''Docstring
    '''
    def __init__(self, spde = None,grid= None, parameters = None,ani = True, ha = True, bc = 3, Q0 = None) -> None:
        assert grid is not None
        self.grid = grid
        if spde is not None:
            self.model(spde = spde,grid = self.grid, parameters=parameters,ani = ani, ha = ha, bc = bc,Q0 = Q0)
        else:
            self.mod = None
        self.mvar = None

    #def setGrid(self,x = None, y = None,extend = None) -> None:
    #    self.grid.setGrid(x = x, y = y,extend = extend)
    #    self.setQ()

    def setQ(self,par = None) -> None:
        self.mod.setQ(par=par)
        
    def model(self,spde=None,grid=None, parameters = None,ani = True, ha = True, bc = 3,Q0 = None) -> None:
        """model _summary_

        Parameters
        ----------
        model : str or int
            Can be either "whittle-matern" or 1, or "advection-diffusion" or 2, "var-advection-diffusion" or 3
        parameters : np.ndarray, optional
            Parameters for the model, by default None
        grid : Grid, optional
            Grid object, by default None
        ha : bool, optional
            Use half angles diffusion, by default True
        bc : int, optional
            Boundary conditions, by default 3
        """
        assert spde is not None and grid is not None
        self.mod = spde_init(model = spde, grid = grid, parameters = parameters,ani = ani, ha = ha, bc = bc, Q0 = Q0)
        self.spde_type = self.mod.type
        self.optim = Optimize(self.mod.logLike)
            
            
    def fit(self,data: np.ndarray ,**kwargs):
        """fit spde model to data

        Parameters
        ----------
        data : np.ndarray
            Data for the model to be fitted to
        x0 : np.ndarray, optional
            Initial guess for the parameters, by default defined by each spde model
        verbose : bool, optional
            Print progress, by default False
        
            
        """
        assert self.mod is not None
        x0 = self.mod.initFit(data,**kwargs)
        if kwargs.get("x0") is None:
            kwargs['x0'] = x0
        if kwargs.get("verbose") is not None:
            kwargs["print"] = self.mod.print
        res = self.optim.fit(**kwargs)
        self.setQ(par = res['x'])
        return(res)

    def sample(self, n = 1,simple = False) -> np.ndarray:
        assert self.mod is not None
        return(self.mod.sample(n = n,simple = simple))
        
    
    def qinv(self,simple = False):
        if simple:
            z = self.sample(n = 1000, simple = True)
            self.mvar = z.var(axis = 1)
        else:
            import rpy2.robjects as robj
            import os
            tmp = os.path.dirname(__file__)
            robj.r.source(+'rqinv.R')
            tshape = self.mod.Q.shape
            Q = self.mod.Q.tocoo()
            r = Q.row
            c = Q.col
            v = Q.data
            tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
            self.pQinv = sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape)
            self.mvar = self.pQinv.diagonal()

    def getPars(self) -> np.ndarray:
        return(self.mod.getPars())

    def plot(self):
        pass