import numpy as np
from scipy import sparse
import ctypes
import os
from sksparse.cholmod import cholesky

class WhittleMatern2D:
    """ Whittle Matern _summary_
    
    Parameters = log kappa^2, log gamma, log tau
    """
    def __init__(self,grid,par=None,bc = 3) -> None:
        self.grid = grid
        self.type = "whittle-matern-isotropic-2D-bc%d"%(bc)
        self.Q = None
        self.Q_fac = None
        self.data = None
        self.r = None
        self.S = None
        self.bc = bc
        self.AHnew = None
        self.Awnew = None
        if par is None: 
            par = np.array([-1,-1,np.log(100)],dtype = "float64")
            self.setPars(par)
        else:
            self.setQ(par = par)
    
    def getPars(self) -> np.ndarray:
        return(np.hstack([self.kappa,self.gamma,self.tau],dtype = "float64"))
    
    def setPars(self,par) -> None:
        self.kappa = par[0]
        self.gamma = par[1]
        self.tau = par[2]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))

    def initFit(self,data, **kwargs):
        assert data.shape[0] <= self.grid.Ns
        par = np.array([-1,-1,np.log(100)],dtype = "float64")
        self.data = data
        if self.data.ndim == 2:
            self.r = self.data.shape[1]
        else:
            self.r = 1
        self.S = self.grid.getS(idxs = kwargs.get("idx"))
        return(par)


    def setQ(self,par = None,S = None):
        if par is None:
            par = self.getPars()
        else:
            self.setPars(par)
        self.Q, self.Q_fac = self.makeQ(par = par, grad = False)
        self.S = self.grid.getS()


    def makeQ(self, par, grad = True):
        Ns = self.grid.Ns
        Dv = self.grid.Dv
        iDv = self.grid.iDv
        # parameters
        kappa = np.exp(par[0])
        gamma = np.exp(par[1])
        # components
        Hs = gamma*np.eye(2)
        Dk =  kappa*sparse.eye(Ns) 
        A =  Dv@Dk - self.Ah(Hs)
        Q = A.transpose()@iDv@A
        # gradient
        Q_fac = cholesky(Q)
        if grad:
            dQ = []
            # log kappa 2
            dA = (Dv@Dk).tocsc()
            dQ.append(dA.T@iDv@A + A.T@iDv@dA)
            # log gamma
            dHs = gamma*np.eye(2) 
            dA = - self.Ah(dHs)
            A =  Dv@Dk - self.Ah(Hs)
            dQ.append(dA.T@iDv@A + A.T@iDv@dA)
            return(Q, Q_fac, dQ)
        else:
            return(Q, Q_fac)
        
    def print(self,par):
        return("| \u03BA = %2.2f"%(np.exp(par[0])) +  ", \u03B3 = %2.2f"%(np.exp(par[1]))  +  ",\u03C4 = %2.2f"%(np.exp(par[2])))
    
    def logLike(self, par, nh1 = 100, grad = True):
        if grad:
            data  = self.data
            tau = np.exp(par[-1])
            Q, Q_fac, dQ = self.makeQ(par = par, grad = True)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac = cholesky(Q_c)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            
            vtmp = (2*np.random.randint(1,3,self.grid.n*nh1)-3).reshape(self.grid.n,nh1)
            TrQ = Q_fac.solve_A(vtmp)
            TrQc = Q_c_fac.solve_A(vtmp)
            g_par = np.zeros(par.size)
            for i in range(par.size-1):
                dQmu_c = dQ[i]@mu_c
                g_par[i]= ((1/2*((TrQ - TrQc)*(dQ[i]@vtmp)).sum()*self.r/nh1 - 1/2*(mu_c*(dQmu_c)).sum()))
            g_par[-1] = self.S.shape[0]*self.r/2 - 1/2*(TrQc*(self.S.T@self.S*tau@vtmp)).sum()*self.r/nh1 - tau/2*((data - self.S@mu_c)**2).sum()
            jac =  -g_par/(self.S.shape[0]*self.r)
            like =  -like/(self.S.shape[0]*self.r)
            return((like,jac))
        else:
            data  = self.data
            tau = np.exp(par[-1])
            Q, Q_fac = self.makeQ(par = par, grad = False)
            Q_c = Q + self.S.T@self.S*tau
            Q_c_fac= cholesky(Q_c)
            mu_c = Q_c_fac.solve_A(self.S.T@data*tau)
            if self.r == 1:
                data = data.reshape(-1,1)
                mu_c = mu_c.reshape(-1,1)
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*np.log(tau)/2 - 1/2*Q_c_fac.logdet()*self.r - 1/2*(mu_c*(Q@mu_c)).sum() - tau/2*((data-self.S@mu_c)**2).sum()
            like =  -like/(self.S.shape[0]*self.r)
            return(like)
    
    def Ah(self, Hs) -> sparse.csc_matrix:
        if self.AHnew is None:
            self.setClib()
        M, N = self.grid.shape
        Hs = np.array(Hs,dtype = "float64")
        obj = self.AHnew(M, N, Hs, self.grid.hx, self.grid.hy)
        row = self.AHrow(obj)
        col = self.AHcol(obj)
        val = self.AHval(obj)

        rem = row != (M*N)
        row = row[rem]
        col = col[rem]
        val = val[rem]
        res = sparse.csc_matrix((val, (row, col)), shape=(M*N, M*N))
        self.AHdel(obj)
        return(res)
        
    def setClib(self) -> None:
        tmp = os.path.dirname(__file__)
            
        M, N = self.grid.shape
        
        if not os.path.exists(tmp + '/ccode/lib_AcH_2D_b%d.so'%(self.bc)):
            os.system('g++ -c -fPIC %s/ccode/AcH_2D_b%d.cpp -o %s/ccode/AcH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            import platform
            if  'Windows' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_AcH_2D_b%d.so -o %s/ccode/lib_AcH_2D_b%d.so %s/ccode/AcH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc)) # not tested
            elif  'Linux' in platform.system():
                os.system('g++ -shared -W1, %s/ccode/lib_AcH_2D_b%d.so -o %s/ccode/lib_AcH_2D_b%d.so %s/ccode/AcH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc,tmp,self.bc))
            else:
                os.system('g++ -shared -o %s/ccode/lib_AcH_2D_b%d.so %s/ccode/AcH_2D_b%d.o'%(tmp,self.bc,tmp,self.bc))
            os.system('rm %s/ccode/AcH_2D_b%d.o'%(tmp,self.bc))
        self.libAh = ctypes.cdll.LoadLibrary('%s/ccode/lib_AcH_2D_b%d.so'%(tmp,self.bc))
        self.AHnew = self.libAh.AH_new
        self.AHnew.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,shape = (2,2)), ctypes.c_double,ctypes.c_double]
        self.AHnew.restype = ctypes.c_void_p
        self.AHrow = self.libAh.AH_Row
        self.AHrow.argtypes = [ctypes.c_void_p]
        self.AHrow.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*9,))
        self.AHcol = self.libAh.AH_Col
        self.AHcol.argtypes = [ctypes.c_void_p]
        self.AHcol.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*9,))
        self.AHval = self.libAh.AH_Val
        self.AHval.argtypes = [ctypes.c_void_p]
        self.AHval.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*9,))
        self.AHdel = self.libAh.AH_delete
        self.AHdel.argtypes = [ctypes.c_void_p]
        self.AHdel.restype = None
