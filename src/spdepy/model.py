import numpy as np
from scipy import sparse
from spdepy.spdes import spde_init
from spdepy.optim import Optimize
from sksparse.cholmod import cholesky

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
        self.Q = None
        self.mvar = None
        self.mu = np.zeros(self.grid.shape)
        self.useCov = False
        self.sigmas = np.log(np.array([0.01,140]))

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

    def sample(self, n = 1, simple = False) -> np.ndarray:
        #self.setModel(mu = self.mu,sigmas = self.sigmas, useCov = self.useCov)
        if not self.useCov:
            z = np.random.normal(size = self.Q.shape[0]*n).reshape(self.Q.shape[0],n)
            self.Q_fac = cholesky(self.Q)
            data = self.grid.getS()@(self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False))+self.mu[:,np.newaxis]) 
        else:
            z = np.random.normal(size = self.Q.shape[0]*n).reshape(self.Q.shape[0],n)
            self.Q_fac = cholesky(self.Q)
            data = self.grid.getS()@(self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False))+self.mu[:,np.newaxis]) 
        if not simple:
            data += self.grid.getS()[:,:np.prod(self.grid.shape)]@z[:np.prod(self.grid.shape)]*1/np.sqrt(self.tau)
        return(data)
    
    def qinv(self,simple = False):
        if simple:
            z = self.sample(n = 1000)
            self.mvar = z.var(axis = 1)
        else:
            if self.Q is None:
                self.setModel()
            import rpy2.robjects as robj
            import os
            tmp = os.path.dirname(__file__)
            robj.r.source(tmp+'/rqinv.R')
            if not self.useCov:
                Q = self.Q.tocoo()
                tshape = self.Q.shape
                r = Q.row
                c = Q.col
                v = Q.data
                tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
                self.pQinv = sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape)
                self.mvar = self.pQinv.diagonal()
            else:
                Q = (self.grid.getS()@self.Q@self.grid.getS().T).tocoo()
                tshape = self.Q.shape
                r = Q.row
                c = Q.col
                v = Q.data
                tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
                self.pQinv = sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape)
                self.mvar = self.pQinv.diagonal()
            
    def update(self, y, idx, tau = None):
        if tau is None:
            tau = self.tau
        S = self.grid.getS(idx)
        self.Q = self.Q + S.transpose()@S*tau
        self.Q_fac = cholesky(self.Q)
        tmp = self.Q_fac.solve_A(S.T@(y-S@self.mu))*tau
        self.mu = self.mu + tmp
        
    def setModel(self, mu = None, sigmas = None, useCov = None):
        self.useCov = useCov if useCov is not None else self.useCov
        if useCov:
            self.sigmas = sigmas if sigmas is not None else self.sigmas
            if hasattr(self.sigmas, "__len__"):
                self.grid.addCov(mu)
                self.Q = sparse.block_diag([self.mod.Q.copy(),np.diag(np.exp(self.sigmas))]).tocsc()
                self.mu = np.zeros(self.Q.shape[0])
                self.mu[-2:] = [0,mu.max()]
                self.tau = np.exp(self.mod.tau)
            else:
                self.grid.addInt()
                self.Q = sparse.block_diag([self.mod.Q.copy(),np.diag([np.exp(self.sigmas)])]).tocsc()
                S = self.grid.getS()
                self.mu = np.zeros(S.shape[1])
                self.mu[-1] = 0
                self.mu[:-1] = S[:,:-1].T@mu
                self.tau = np.exp(self.mod.tau)
        else:
            self.mu = np.zeros(self.mod.Q.shape[0]) if mu is None else self.grid.getS().T@mu
            self.Q = self.mod.Q.copy().tocsc()
            self.tau = np.exp(self.mod.tau)

    def getPars(self) -> np.ndarray:
        return(self.mod.getPars())

    def plot(self):
        value = 1
        self.grid.plot(value)